"""双音箱仲裁引擎测试：暴力参考实现对拍 + 流式分块追加 + 全量校验路径。

运行：python -I -m pytest tests/ -q
"""

from __future__ import annotations

import random

import pytest

from app.dual_engine import (
    MAX_COOLDOWN,
    MAX_FRAMES,
    MAX_LEAD,
    MAX_W,
    Candidate,
    DualError,
    DualSession,
)


# ---------------- 暴力参考实现：一次性拿到两路完整分数，逐帧模拟 ----------------


def brute_run(
    s0: list[int],
    s1: list[int],
    t0: int,
    t1: int,
    c0: int,
    c1: int,
    w: int,
) -> list[dict]:
    """与需求逐条对应的参考实现；两路必须等长。"""
    assert len(s0) == len(s1)
    n = len(s0)
    avail = [0, 0]  # 下一可触发帧号
    scores = (s0, s1)
    thresholds = (t0, t1)
    cooldowns = (c0, c1)

    answers: list[dict] = []
    frame = 0
    while frame < n:
        # 找任一台未冷却且达阈的首帧
        start = None
        f = frame
        while f < n:
            hit0 = f >= avail[0] and scores[0][f] >= thresholds[0]
            hit1 = f >= avail[1] and scores[1][f] >= thresholds[1]
            if hit0 or hit1:
                start = f
                break
            f += 1
        if start is None:
            break
        # 收集窗内候选（只在共同帧上，参考实现两路等长）
        cands: list[Candidate] = []
        end_full = min(n - 1, start + w)
        for g in range(start, end_full + 1):
            for ch in (0, 1):
                if g >= avail[ch] and scores[ch][g] >= thresholds[ch]:
                    cands.append(Candidate(ch, g, scores[ch][g]))
        best = max(cands, key=lambda c: (c.score, -c.frame, -c.channel))
        end = end_full  # 完整窗为 start+w，末窗不足时为最后一帧
        answers.append(
            {
                "speaker": best.channel + 1,
                "selected_frame": best.frame,
                "answer_frame": end,
                "window_start": start,
                "window_end": end,
                "score": best.score,
            }
        )
        avail[best.channel] = end + cooldowns[best.channel] + 1
        frame = end + 1
    return answers


def answers_of(session: DualSession) -> list[dict]:
    return [
        {
            "speaker": a["speaker"],
            "selected_frame": a["selected_frame"],
            "answer_frame": a["answer_frame"],
            "window_start": a["window_start"],
            "window_end": a["window_end"],
            "score": a["score"],
        }
        for a in session.answers
    ]


# ---------------- 基本规则 ----------------


def test_handcrafted_pick_higher_score():
    # 帧 0 两台都达标：一号 80、二号 90 -> 二号胜，在窗末（W=2 -> 帧 2）应答
    s0 = [80, 0, 0, 0, 0, 0]
    s1 = [90, 0, 0, 0, 0, 0]
    sess = DualSession(thresholds=(70, 70), cooldowns=(2, 2), w=2)
    sess.append_block(0, 0, s0)
    sess.append_block(1, 0, s1)
    sess.finish()
    assert answers_of(sess) == [
        {
            "speaker": 2,
            "selected_frame": 0,
            "answer_frame": 2,
            "window_start": 0,
            "window_end": 2,
            "score": 90,
        }
    ]


def test_tie_prefers_earlier_then_speaker_one():
    # 窗 [0..3]：帧 1 二号 85；帧 2 一号 85（同分取较早 -> 二号）
    s0 = [0, 0, 85, 0]
    s1 = [0, 85, 0, 0]
    sess = DualSession(thresholds=(80, 80), cooldowns=(1, 1), w=3)
    sess.append_block(0, 0, s0)
    sess.append_block(1, 0, s1)
    sess.finish()
    assert sess.answers[0]["speaker"] == 2
    assert sess.answers[0]["selected_frame"] == 1

    # 同分且同帧 -> 一号音箱
    s0 = [85, 0, 0, 0]
    s1 = [85, 0, 0, 0]
    sess = DualSession(thresholds=(80, 80), cooldowns=(1, 1), w=3)
    sess.append_block(0, 0, s0)
    sess.append_block(1, 0, s1)
    sess.finish()
    assert sess.answers[0]["speaker"] == 1
    assert sess.answers[0]["selected_frame"] == 0


def test_winner_cools_loser_keeps_progress():
    # 一号帧 0 达标应答（窗末帧 0，W=0），冷却 2 帧 -> 帧 1、2 冷却，帧 3 起可用。
    # 二号帧 2 自身达标时一号仍在冷却，二号在帧 2 开窗并应答。
    # 之后一号在帧 3 恢复可用且达阈，再开一轮（末窗按最后一帧结算）。
    s0 = [90, 0, 0, 90]
    s1 = [0, 0, 80, 0]
    sess = DualSession(thresholds=(80, 80), cooldowns=(2, 2), w=0)
    sess.append_block(0, 0, s0)
    sess.append_block(1, 0, s1)
    sess.finish()
    got = answers_of(sess)
    assert [(a["speaker"], a["selected_frame"], a["answer_frame"]) for a in got] == [
        (1, 0, 0),
        (2, 2, 2),
        (1, 3, 3),
    ]


def test_cooldown_state_uses_avail_frame():
    # 胜者冷却进度的快照值 = max(0, avail - frontier)
    s0 = [90, 0, 0, 0, 0]
    s1 = [0, 0, 0, 0, 0]
    sess = DualSession(thresholds=(80, 80), cooldowns=(3, 3), w=1)
    sess.append_block(0, 0, s0)
    sess.append_block(1, 0, s1)
    # 窗 [0..1] 结算后继续扫到末尾，frontier=5；一号 avail=1+3+1=5 -> 剩余 5-5=0
    snap = sess.snapshot()
    assert snap["frontier"] == 5
    assert snap["cooldown_remaining"] == [0, 0]
    # 直接核验内部 avail：帧 2、3、4 处于冷却
    assert sess.avail[0] == 5


def test_last_incomplete_window_settles_on_last_frame():
    # W=50 但只有 4 帧：末窗按最后一帧（帧 3）结算
    s0 = [90, 0, 0, 0]
    s1 = [0, 0, 0, 0]
    sess = DualSession(thresholds=(80, 80), cooldowns=(1, 1), w=50)
    sess.append_block(0, 0, s0)
    sess.append_block(1, 0, s1)
    assert sess.answers == []  # 未结束前，窗末未到，不结算
    sess.finish()
    assert len(sess.answers) == 1
    assert sess.answers[0]["window_end"] == 3
    assert sess.answers[0]["answer_frame"] == 3


def test_unequal_lengths_cannot_finish_then_resume():
    s0 = [90, 0, 0, 0, 0]
    s1 = [0, 0, 0]
    sess = DualSession(thresholds=(80, 80), cooldowns=(1, 1), w=1)
    sess.append_block(0, 0, s0)
    sess.append_block(1, 0, s1)
    with pytest.raises(DualError, match="两路帧数不一致"):
        sess.finish()
    assert sess.finished is False
    # 继续追加较短一路，成功结束
    sess.append_block(1, 3, [0, 0])
    sess.finish()
    assert sess.finished is True
    # 成功后拒收新块
    with pytest.raises(DualError, match="回放已结束"):
        sess.append_block(0, 5, [1])


# ---------------- 分块校验：整块退回 ----------------


def test_first_block_must_start_at_zero():
    sess = DualSession(thresholds=(1, 1), cooldowns=(0, 0), w=0)
    with pytest.raises(DualError, match="起始帧应为 0"):
        sess.append_block(0, 1, [1])
    assert sess.snapshot()["lengths"] == [0, 0]
    assert sess.blocks == []


def test_blocks_must_continue_own_channel_numbering():
    sess = DualSession(thresholds=(1, 1), cooldowns=(0, 0), w=0)
    sess.append_block(0, 0, [1, 2])
    sess.append_block(1, 0, [3, 4])
    with pytest.raises(DualError, match="起始帧应为 2"):
        sess.append_block(0, 3, [5])  # 跳过帧 2
    # 整块退回：一号仍是 2 帧，且没有新块记录
    assert sess.snapshot()["lengths"] == [2, 2]
    assert len(sess.blocks) == 2
    # 正确接续被接受
    sess.append_block(0, 2, [5])
    assert sess.snapshot()["lengths"] == [3, 2]


def test_max_lead_enforced_and_atomic_rollback():
    sess = DualSession(thresholds=(100, 100), cooldowns=(0, 0), w=0)
    # 一号先放 MAX_LEAD 帧（二号仍为空，差恰好 4096，允许）
    sess.append_block(0, 0, [0] * MAX_LEAD)
    assert sess.snapshot()["lengths"] == [MAX_LEAD, 0]
    # 再追加一帧 -> 差 4097，拒绝并整块回滚
    with pytest.raises(DualError, match="超过单路最大领先"):
        sess.append_block(0, MAX_LEAD, [0])
    assert sess.snapshot()["lengths"] == [MAX_LEAD, 0]
    assert len(sess.blocks) == 1
    # 二号追上 10 帧后，一号可继续推进到领先 4096
    sess.append_block(1, 0, [0] * 10)
    sess.append_block(0, MAX_LEAD, [0] * 10)
    assert sess.snapshot()["lengths"] == [MAX_LEAD + 10, 10]


def test_max_frames_per_channel():
    sess = DualSession(thresholds=(100, 100), cooldowns=(0, 0), w=0)
    # 两路交错追加（每块 4000，不触发领先限制），一直顶到 20 万上限
    chunk = 4_000
    for start in range(0, MAX_FRAMES, chunk):
        size = min(chunk, MAX_FRAMES - start)
        sess.append_block(0, start, [0] * size)
        sess.append_block(1, start, [0] * size)
    assert sess.snapshot()["lengths"] == [MAX_FRAMES, MAX_FRAMES]
    with pytest.raises(DualError, match="超过每路上限"):
        sess.append_block(0, MAX_FRAMES, [0])


def test_block_input_validation():
    sess = DualSession(thresholds=(1, 1), cooldowns=(0, 0), w=0)
    with pytest.raises(DualError):
        sess.append_block(0, 0, [])
    with pytest.raises(DualError):
        sess.append_block(0, 0, [1, 2.0])  # 非整数
    with pytest.raises(DualError):
        sess.append_block(2, 0, [1])  # 音箱编号非法
    with pytest.raises(DualError):
        sess.append_block(0, -1, [1])


def test_config_validation():
    with pytest.raises(DualError, match="冷却帧数"):
        DualSession(thresholds=(1, 1), cooldowns=(MAX_COOLDOWN + 1, 0), w=0)
    with pytest.raises(DualError, match="等待帧数"):
        DualSession(thresholds=(1, 1), cooldowns=(0, 0), w=MAX_W + 1)
    with pytest.raises(DualError):
        DualSession(thresholds=(1, 1), cooldowns=(0, 0), w=-1)
    with pytest.raises(DualError):
        DualSession(thresholds=(1, 1.5), cooldowns=(0, 0), w=0)


# ---------------- 不重扫：frontier 单调，追加只处理新帧 ----------------


def test_no_rescan_frontier_monotonic_and_cumulative():
    rng = random.Random(11)
    sess = DualSession(thresholds=(50, 50), cooldowns=(3, 3), w=4)
    max_seen_frontier = 0
    # 两路交替小块追加，制造一路暂时领先
    plan = [(0, 8), (1, 5), (0, 7), (1, 10), (0, 3), (1, 3)]
    for ch, size in plan:
        start = len(sess.scores[ch])
        before = sess.frontier
        sess.append_block(ch, start, [rng.randint(0, 100) for _ in range(size)])
        assert sess.frontier >= before  # 单调不减
        max_seen_frontier = max(max_seen_frontier, sess.frontier)
        # 已处理区不超过两路共同长度
        assert sess.frontier <= min(len(sess.scores[0]), len(sess.scores[1]))
    assert max_seen_frontier > 0
    # 把两路补齐到等长后结束，结果与一次性暴力模拟一致
    target = max(len(sess.scores[0]), len(sess.scores[1])) + 4
    for ch in (0, 1):
        cur = len(sess.scores[ch])
        if cur < target:
            sess.append_block(ch, cur, [rng.randint(0, 100) for _ in range(target - cur)])
    sess.finish()
    want = brute_run(sess.scores[0], sess.scores[1], 50, 50, 3, 3, 4)
    assert answers_of(sess) == want


# ---------------- 随机对拍：流式分块 vs 暴力参考 ----------------


def test_property_random_vs_bruteforce():
    rng = random.Random(20261122)
    for _ in range(300):
        n = rng.randint(1, 60)
        t0, t1 = rng.randint(2, 12), rng.randint(2, 12)
        c0, c1 = rng.randint(0, 6), rng.randint(0, 6)
        w = rng.randint(0, 8)
        full0 = [rng.randint(0, 15) for _ in range(n)]
        full1 = [rng.randint(0, 15) for _ in range(n)]

        sess = DualSession(thresholds=(t0, t1), cooldowns=(c0, c1), w=w)
        # 随机把两路切成小块、交错追加，允许一路暂时领先
        cursors = [0, 0]
        while cursors[0] < n or cursors[1] < n:
            ch = rng.randint(0, 1)
            if cursors[ch] >= n:
                ch = 1 - ch
            size = rng.randint(1, min(8, n - cursors[ch]))
            # 若会违反 4096 限制（小 n 下不会），缩短到合法长度
            other = len(sess.scores[1 - ch])
            if abs((cursors[ch] + size) - other) > MAX_LEAD:
                size = max(1, other + MAX_LEAD - cursors[ch])
            data = (full0 if ch == 0 else full1)[cursors[ch] : cursors[ch] + size]
            sess.append_block(ch, cursors[ch], data)
            cursors[ch] += size
            # 引擎在处理窗末的下一帧时结算；暴力法对截断前缀的悬窗会提前按末帧结算
            # （窗长不足 W+1），这类悬窗流式侧要等后续追加/结束才结算，故排除。
            common = min(len(sess.scores[0]), len(sess.scores[1]))
            ref = brute_run(full0[:common], full1[:common], t0, t1, c0, c1, w)
            settled = [a for a in ref if a["window_end"] - a["window_start"] == w]
            assert answers_of(sess) == settled
        sess.finish()
        assert answers_of(sess) == brute_run(full0, full1, t0, t1, c0, c1, w)


def test_property_incremental_matches_one_shot_prefixes():
    # 每追加一块，已确定的应答必须与"在当前共同前缀上整体重算"完全一致
    rng = random.Random(99)
    for _ in range(120):
        n = rng.randint(1, 50)
        t0, t1 = rng.randint(0, 10), rng.randint(0, 10)
        c0, c1 = rng.randint(0, 5), rng.randint(0, 5)
        w = rng.randint(0, 6)
        full0 = [rng.randint(0, 10) for _ in range(n)]
        full1 = [rng.randint(0, 10) for _ in range(n)]
        sess = DualSession(thresholds=(t0, t1), cooldowns=(c0, c1), w=w)
        cut = 0
        order = [0] * n + [1] * n
        rng.shuffle(order)
        lengths = [0, 0]
        for ch in order:
            if lengths[ch] >= n:
                continue
            data = (full0 if ch == 0 else full1)[lengths[ch]]
            sess.append_block(ch, lengths[ch], [data])
            lengths[ch] += 1
            common = min(lengths)
            # 暴力法对截断悬窗会提前按末帧结算；只比较窗长完整（W+1 帧）的应答
            ref = brute_run(full0[:common], full1[:common], t0, t1, c0, c1, w)
            settled = [a for a in ref if a["window_end"] - a["window_start"] == w]
            got = answers_of(sess)
            assert got == settled, (lengths, got, settled)
        sess.finish()
        assert answers_of(sess) == brute_run(full0, full1, t0, t1, c0, c1, w)


def test_max_size_performance():
    # 两路各 20 万帧，交错小块追加也必须线性完成
    rng = random.Random(3)
    n = MAX_FRAMES
    sess = DualSession(thresholds=(50, 60), cooldowns=(MAX_COOLDOWN, MAX_COOLDOWN), w=MAX_W)
    import time

    t0 = time.perf_counter()
    chunk = 4_000  # 单块不超过 4096 领先限制
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        sess.append_block(0, start, [rng.randint(0, 100) for _ in range(size)])
        sess.append_block(1, start, [rng.randint(0, 100) for _ in range(size)])
    sess.finish()
    elapsed = time.perf_counter() - t0
    assert sess.frontier == n
    assert elapsed < 8.0
