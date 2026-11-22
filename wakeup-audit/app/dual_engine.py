"""双音箱唤醒仲裁引擎（流式分块追加，不重扫已处理帧）。

场景：两台音箱同时监听同一唤醒词，都可能达标，要求每次唤醒只有一台应答。

输入：两路对齐、同标度的逐帧整数分数（帧号在两路之间对齐），
每路各自的阈值 threshold[ch] 与冷却帧数 cooldown[ch]，以及等待帧数 W。

规则（与需求逐条对应）：
  1. 分数按块追加：每块注明起始帧，首块必须从 0 开始，后续块须接续本路帧号；
     每路最多 MAX_FRAMES 帧；任一路最多领先 MAX_LEAD 帧。校验失败整块退回。
  2. 两台音箱起初都可触发（avail[ch] = 0）。
  3. 只在两路都已收到的帧上推进：从当前位置找第一帧 f，使任一台音箱
     未冷却且分数达阈；以 f 为窗首，纳入 f..f+W 共 W+1 帧。
  4. 窗末（两路都收到 f+W）才结算：候选为窗内"未冷却且达阈"的帧，
     取分数最高者；同分取较早帧；再同分取一号音箱。
  5. 胜者在窗末帧应答，从下一帧起冷却 cooldown[ch] 帧；另一台沿用原冷却进度。
  6. 下一轮从窗末后一帧开始；未满的末窗在结束回放时按最后一帧结算。
  7. 两路等长才允许结束；结束失败可继续追加，成功后拒收新块。

实现要点：frontier 单调递增，每个共同帧只处理一次，追加新块不会重扫旧帧；
冷却用"下一可触发帧号 avail[ch]"表示，胜者置为 窗末+cooldown+1，
负者完全不动，天然实现"沿用原来的冷却进度"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_FRAMES = 200_000
MAX_COOLDOWN = 2000
MAX_W = 50
MAX_LEAD = 4096


class DualError(ValueError):
    """输入不合法或状态冲突时抛出，message 为可直接展示的中文。"""


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass
class Candidate:
    channel: int  # 0 = 一号音箱，1 = 二号音箱
    frame: int
    score: int


@dataclass
class DualSession:
    thresholds: tuple[int, int]
    cooldowns: tuple[int, int]
    w: int

    # 每路完整分数（仅追加，不重排）
    scores: list[list[int]] = field(default_factory=lambda: [[], []])
    # 已接受块的审计信息
    blocks: list[dict] = field(default_factory=list)
    # 应答记录（每轮一条）
    answers: list[dict] = field(default_factory=list)

    # ---- 流式状态 ----
    # 下一待处理的共同帧号；[0, frontier) 的共同帧均已处理，永不重扫
    frontier: int = 0
    # 每台音箱下一可触发帧号（未冷却且达阈判定用）
    avail: list[int] = field(default_factory=lambda: [0, 0])
    # 当前开窗状态
    open_window: bool = False
    window_start: int = 0
    # 窗内每台音箱已收集的候选（帧号随 frontier 单调加入，不回看旧帧）
    pending: list[list[Candidate]] = field(default_factory=lambda: [[], []])

    finished: bool = False

    # ---------------- 配置校验 ----------------

    def __post_init__(self) -> None:
        if not all(_is_int(v) for v in self.thresholds):
            raise DualError("两台音箱的阈值都必须是整数。")
        if not all(_is_int(v) for v in self.cooldowns):
            raise DualError("两台音箱的冷却帧数都必须是整数。")
        if not _is_int(self.w):
            raise DualError("等待帧数 W 必须是整数。")
        for ch, cd in enumerate(self.cooldowns):
            if not 0 <= cd <= MAX_COOLDOWN:
                raise DualError(f"{'一二'[ch]}号音箱冷却帧数必须在 0～{MAX_COOLDOWN} 之间。")
        if not 0 <= self.w <= MAX_W:
            raise DualError(f"等待帧数 W 必须在 0～{MAX_W} 之间。")

    # ---------------- 分块追加 ----------------

    def append_block(self, channel: int, start_frame: int, block: list[int]) -> dict:
        """校验并追加一块分数，然后推进流式仲裁。任何校验失败都整块退回。"""
        if self.finished:
            raise DualError("回放已结束，拒收新块。")
        if not _is_int(channel) or channel not in (0, 1):
            raise DualError("音箱编号只能是 1 或 2。")
        if not _is_int(start_frame) or start_frame < 0:
            raise DualError("起始帧必须是非负整数。")
        if not isinstance(block, list) or not block:
            raise DualError("分数块不能为空。")
        if not all(_is_int(v) for v in block):
            raise DualError("分数必须全部为整数。")

        expected = len(self.scores[channel])
        if start_frame != expected:
            raise DualError(
                f"{'一二'[channel]}号音箱块起始帧应为 {expected}（本路当前帧数），"
                f"实际填写 {start_frame}，整块退回。"
            )
        if expected + len(block) > MAX_FRAMES:
            raise DualError(
                f"{'一二'[channel]}号音箱追加后将达 {expected + len(block)} 帧，"
                f"超过每路上限 {MAX_FRAMES}，整块退回。"
            )

        # 先在影子长度上检查领先约束，违反则不动任何状态
        new_len = expected + len(block)
        other_len = len(self.scores[1 - channel])
        if abs(new_len - other_len) > MAX_LEAD:
            raise DualError(
                f"追加后两路帧差为 {abs(new_len - other_len)} 帧，"
                f"超过单路最大领先 {MAX_LEAD} 帧，整块退回。"
            )

        # 全部校验通过才真正落块
        self.scores[channel].extend(block)
        self.blocks.append(
            {
                "channel": channel + 1,
                "start_frame": start_frame,
                "frame_count": len(block),
                "end_frame": start_frame + len(block) - 1,
            }
        )

        answers_before = len(self.answers)
        self._advance()
        return self.snapshot(new_answers=len(self.answers) - answers_before)

    # ---------------- 流式仲裁（每帧只处理一次） ----------------

    def _ready(self, ch: int, frame: int) -> bool:
        return frame >= self.avail[ch] and self.scores[ch][frame] >= self.thresholds[ch]

    def _advance(self) -> None:
        """处理两路都已收到、且尚未处理的共同帧；frontier 只增不减。"""
        common = min(len(self.scores[0]), len(self.scores[1]))
        while self.frontier < common:
            frame = self.frontier

            if not self.open_window:
                # 找任一台未冷却且达阈的首帧作为窗首
                if self._ready(0, frame) or self._ready(1, frame):
                    self.open_window = True
                    self.window_start = frame
                    self.pending = [[], []]

            if self.open_window:
                for ch in (0, 1):
                    if self._ready(ch, frame):
                        self.pending[ch].append(
                            Candidate(ch, frame, self.scores[ch][frame])
                        )
                window_end = self.window_start + self.w
                if frame >= window_end:
                    # 两路都已收到窗末（frame 本身就是共同帧），结算
                    self._settle(window_end)
                    self.open_window = False
                    self.frontier = window_end + 1
                    continue

            self.frontier = frame + 1

    def _settle(self, window_end: int) -> None:
        candidates = [c for lst in self.pending for c in lst]
        # 开窗首帧必入候选，故不会为空
        best = max(candidates, key=lambda c: (c.score, -c.frame, -c.channel))
        self.answers.append(
            {
                "round": len(self.answers) + 1,
                "speaker": best.channel + 1,
                "selected_frame": best.frame,
                "answer_frame": window_end,
                "window_start": self.window_start,
                "window_end": window_end,
                "score": best.score,
            }
        )
        # 胜者从窗末下一帧起冷却 cooldown 帧；另一台 avail 不动，沿用原进度
        self.avail[best.channel] = window_end + self.cooldowns[best.channel] + 1

    # ---------------- 结束回放 ----------------

    def finish(self) -> dict:
        if self.finished:
            raise DualError("回放已结束，拒收新操作。")
        n0, n1 = len(self.scores[0]), len(self.scores[1])
        if n0 != n1:
            raise DualError(
                f"两路帧数不一致（一号 {n0} 帧、二号 {n1} 帧），"
                "无法结束回放：请继续追加较短一路后重试。"
            )
        if self.open_window:
            # 未满的末窗按最后一帧结算
            last = n0 - 1
            self._settle(last)
            self.open_window = False
        self.finished = True
        return self.snapshot(new_answers=0)

    # ---------------- 对外快照 ----------------

    def snapshot(self, new_answers: int = 0) -> dict:
        return {
            "finished": self.finished,
            "config": {
                "thresholds": list(self.thresholds),
                "cooldowns": list(self.cooldowns),
                "w": self.w,
            },
            "lengths": [len(self.scores[0]), len(self.scores[1])],
            "frontier": self.frontier,
            "next_start": [len(self.scores[0]), len(self.scores[1])],
            "cooldown_remaining": [max(0, self.avail[ch] - self.frontier) for ch in (0, 1)],
            "blocks": list(self.blocks),
            "answers": list(self.answers),
            "answer_count": len(self.answers),
            "new_answers": new_answers,
        }
