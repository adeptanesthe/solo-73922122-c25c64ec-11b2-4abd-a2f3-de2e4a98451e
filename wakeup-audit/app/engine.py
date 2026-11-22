"""核对引擎：根据模型分数、阈值、冷却帧数和回执，重建缺报区间内的触发帧。

触发规则（确定性）：
  未处于冷却的帧，一旦分数达到阈值即触发；触发后接下来 cooldown 帧进入冷却。
  录制开始时可能仍在冷却，剩余帧数 r 可取 0..cooldown，每种 r 是一种"解释"。

算法（禁止对每种 r 逐帧重扫分数）：
  1. 一次 O(N) 倒序扫描，构造 next_cross[i] = 从帧 i 起第一个分数达阈的帧。
  2. 状态 s（下一可触发帧）到触发序列是一条函数链：
         s -> t = next_cross[s] -> s' = t + cooldown + 1 -> ...
     用记忆化保存链上状态的触发序列，所有 r 的链共享后缀，
     全部链的总长度 O(N + cooldown)，即整体 O(N + cooldown) 时间/空间。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_FRAMES = 200_000
MAX_COOLDOWN = 2000


class VerifyError(ValueError):
    """输入不合法或核对冲突时抛出，message 为可直接展示的中文。"""


def build_next_cross(scores: list[int], threshold: int) -> list[int]:
    """next_cross[i] = 从帧 i（含）开始第一个分数 >= threshold 的帧号，不存在则为 n。"""
    n = len(scores)
    nxt = [n] * (n + 1)
    last = n
    for i in range(n - 1, -1, -1):
        if scores[i] >= threshold:
            last = i
        nxt[i] = last
    return nxt


def build_chain(
    start: int,
    nxt: list[int],
    cooldown: int,
    n: int,
    memo: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    """求从状态 start（下一可触发帧号）开始的全部触发帧，结果按帧号升序。

    迭代式沿函数链走到已知状态，再反向拼接，避免深递归（cooldown=0 时链可长达 N）。
    """
    if start in memo:
        return memo[start]
    path: list[tuple[int, int]] = []
    s = start
    while s < n and s not in memo:
        t = nxt[s]
        if t == n:  # 之后再无达阈帧
            memo[s] = ()
            break
        path.append((s, t))
        s = t + cooldown + 1
    cur = memo.get(s, ())
    while path:
        st, tt = path.pop()
        cur = (tt,) + cur
        memo[st] = cur
    return memo.get(start, ())


@dataclass
class CheckResult:
    status: str  # "ok" 可补记 / "conflict" 冲突
    frame_count: int
    threshold: int
    cooldown: int
    gap_start: int
    gap_end: int
    receipt_count: int
    consistent_rs: list[int] = field(default_factory=list)
    # 缺报区间内、所有一致解释都触发的帧（含已在区间内收到回执的帧）
    backfill: list[dict] = field(default_factory=list)
    # 缺报区间内、仅部分一致解释触发的帧
    pending: list[dict] = field(default_factory=list)
    # 冲突时每种 r 的否决原因
    rejections: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def verify(
    scores: list[int],
    threshold: int,
    cooldown: int,
    gap_start: int,
    gap_end: int,
    receipts: list[int],
) -> CheckResult:
    # ---------- 输入校验 ----------
    if not isinstance(scores, list) or not scores:
        raise VerifyError("分数序列不能为空。")
    n = len(scores)
    if n > MAX_FRAMES:
        raise VerifyError(f"分数帧数最多 {MAX_FRAMES}，当前 {n} 帧。")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in scores):
        raise VerifyError("分数必须全部为整数。")
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise VerifyError("阈值必须为整数。")
    if not isinstance(cooldown, int) or isinstance(cooldown, bool):
        raise VerifyError("冷却帧数必须为整数。")
    if not 0 <= cooldown <= MAX_COOLDOWN:
        raise VerifyError(f"冷却帧数必须在 0～{MAX_COOLDOWN} 之间。")
    if not isinstance(gap_start, int) or not isinstance(gap_end, int):
        raise VerifyError("缺报区间端点必须为整数帧号。")
    if not (0 <= gap_start <= gap_end < n):
        raise VerifyError(
            f"缺报区间非法：需满足 0 <= 起点 <= 终点 < {n}（当前 {gap_start}..{gap_end}）。"
        )
    if len(receipts) != len(set(receipts)):
        raise VerifyError("回执帧号存在重复。")
    if any(not isinstance(f, int) or isinstance(f, bool) or not 0 <= f < n for f in receipts):
        raise VerifyError(f"回执帧号必须是 0..{n - 1} 之间的整数。")

    receipts_sorted = sorted(receipts)
    receipt_set = set(receipts)

    # ---------- O(N) 预处理 + 记忆化跳跃链 ----------
    nxt = build_next_cross(scores, threshold)
    memo: dict[int, tuple[int, ...]] = {n: ()}
    chains: list[tuple[int, tuple[int, ...]]] = []
    for r in range(cooldown + 1):
        chains.append((r, build_chain(r, nxt, cooldown, n, memo)))

    def in_gap(f: int) -> bool:
        return gap_start <= f <= gap_end

    # ---------- 筛选一致解释 ----------
    consistent: list[tuple[int, tuple[int, ...]]] = []
    rejections: list[dict] = []
    for r, tr in chains:
        reason = None
        # 方向一：区间外的触发帧必须恰好是回执帧（区间内触发不受此限）
        for f in tr:
            if not in_gap(f) and f not in receipt_set:
                reason = f"帧 {f} 在缺报区间外触发，但回执中没有该帧"
                break
        if reason is None:
            # 方向二：每张回执（含区间内）都必须在该解释下确实触发
            i = 0
            lt = len(tr)
            for f in receipts_sorted:
                while i < lt and tr[i] < f:
                    i += 1
                if i >= lt or tr[i] != f:
                    if in_gap(f):
                        reason = f"区间内回执帧 {f} 在该解释下不会触发"
                    else:
                        reason = f"回执帧 {f} 在缺报区间外，但该解释下未触发"
                    break
        if reason is None:
            consistent.append((r, tr))
        else:
            rejections.append({"r": r, "reason": reason})

    result = CheckResult(
        status="ok",
        frame_count=n,
        threshold=threshold,
        cooldown=cooldown,
        gap_start=gap_start,
        gap_end=gap_end,
        receipt_count=len(receipts),
    )

    if not consistent:
        result.status = "conflict"
        result.rejections = rejections
        result.stats = {
            "interpretation_count": cooldown + 1,
            "consistent_count": 0,
            "backfill_count": 0,
            "pending_count": 0,
            "memoized_states": len(memo),
        }
        return result

    # ---------- 区间内：交集补记 / 差集待核 ----------
    certain: set[int] | None = None
    possible: set[int] = set()
    for _r, tr in consistent:
        gap_hits = {f for f in tr if in_gap(f)}
        possible |= gap_hits
        certain = gap_hits if certain is None else certain & gap_hits
    assert certain is not None

    for f in sorted(certain):
        result.backfill.append(
            {
                "frame": f,
                "score": scores[f],
                "status": "received" if f in receipt_set else "backfill",
            }
        )
    for f in sorted(possible - certain):
        result.pending.append({"frame": f, "score": scores[f]})

    result.consistent_rs = [r for r, _ in consistent]
    result.stats = {
        "interpretation_count": cooldown + 1,
        "consistent_count": len(consistent),
        "backfill_count": len(result.backfill),
        "backfill_new_count": sum(1 for row in result.backfill if row["status"] == "backfill"),
        "received_in_gap_count": sum(1 for row in result.backfill if row["status"] == "received"),
        "pending_count": len(result.pending),
        "memoized_states": len(memo),
    }
    return result
