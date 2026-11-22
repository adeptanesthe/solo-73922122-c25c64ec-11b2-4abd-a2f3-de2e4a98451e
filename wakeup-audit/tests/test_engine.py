"""引擎对拍测试：暴力逐帧参考实现 vs 记忆化跳跃链实现。

运行：python -I -m pytest tests/ -q   （需 fastapi/httpx 才能跑 test_api.py）
"""

from __future__ import annotations

import random

import pytest

from app.engine import MAX_COOLDOWN, MAX_FRAMES, VerifyError, verify


def brute_triggers(scores: list[int], threshold: int, cooldown: int, r0: int) -> list[int]:
    """参考实现：给定初始冷却剩余 r0，逐帧模拟完整触发序列（仅用于对拍）。"""
    out: list[int] = []
    r = r0
    for i, v in enumerate(scores):
        if r > 0:
            r -= 1
            continue
        if v >= threshold:
            out.append(i)
            r = cooldown
    return out


def brute_verify(scores, threshold, cooldown, gap_start, gap_end, receipts):
    n = len(scores)
    receipt_set = set(receipts)
    consistent_rs, backfill, pending, rejections = [], set(), set(), []
    certain: set[int] | None = None
    possible: set[int] = set()
    for r0 in range(cooldown + 1):
        tr = brute_triggers(scores, threshold, cooldown, r0)
        tr_set = set(tr)
        reason = None
        for f in tr:
            if not (gap_start <= f <= gap_end) and f not in receipt_set:
                reason = f"帧 {f} 在缺报区间外触发，但回执中没有该帧"
                break
        if reason is None:
            for f in receipts:
                if f not in tr_set:
                    reason = f"回执帧 {f} 未触发"
                    break
        if reason is None:
            consistent_rs.append(r0)
            hits = {f for f in tr if gap_start <= f <= gap_end}
            possible |= hits
            certain = hits if certain is None else certain & hits
        else:
            rejections.append((r0, reason))
    if not consistent_rs:
        return {"status": "conflict", "consistent_rs": [], "backfill": [], "pending": [],
                "rejections": rejections}
    return {
        "status": "ok",
        "consistent_rs": consistent_rs,
        "backfill": sorted(certain),
        "pending": sorted(possible - certain),
    }


def engine_summary(result):
    return {
        "status": result.status,
        "consistent_rs": result.consistent_rs,
        "backfill": sorted(row["frame"] for row in result.backfill),
        "pending": sorted(row["frame"] for row in result.pending),
        "rejections": [(row["r"], row["reason"]) for row in result.rejections],
    }


def assert_matches(scores, threshold, cooldown, gs, ge, receipts):
    res = verify(scores, threshold, cooldown, gs, ge, receipts)
    got, want = engine_summary(res), brute_verify(scores, threshold, cooldown, gs, ge, receipts)
    assert got["status"] == want["status"]
    assert got["consistent_rs"] == want["consistent_rs"]
    assert got["backfill"] == want["backfill"]
    assert got["pending"] == want["pending"]
    if got["status"] == "conflict":
        assert len(got["rejections"]) == cooldown + 1
    return got


def test_property_random_small():
    rng = random.Random(20261122)
    for _ in range(400):
        n = rng.randint(1, 40)
        scores = [rng.randint(0, 10) for _ in range(n)]
        threshold = rng.randint(2, 11)
        cooldown = rng.randint(0, 6)
        gs = rng.randint(0, n - 1)
        ge = rng.randint(gs, n - 1)
        # 回执：随机选一个 r0 的真实序列，再随机丢帧（制造缺报）/ 全量
        true_r = rng.randint(0, cooldown)
        true_tr = brute_triggers(scores, threshold, cooldown, true_r)
        keep = [f for f in true_tr if not (gs <= f <= ge and rng.random() < 0.7)]
        if rng.random() < 0.2:
            keep = true_tr[:]  # 完整回执
        receipts = sorted(set(keep))
        assert_matches(scores, threshold, cooldown, gs, ge, receipts)


def test_property_random_medium_with_bad_receipts():
    rng = random.Random(77)
    for _ in range(150):
        n = rng.randint(30, 200)
        scores = [rng.randint(0, 20) for _ in range(n)]
        threshold = rng.randint(5, 22)
        cooldown = rng.randint(0, 30)
        gs = rng.randint(0, n // 2)
        ge = rng.randint(gs, n - 1)
        receipts = sorted(rng.sample(range(n), rng.randint(0, 5)))  # 多为不可行回执 -> 冲突路径
        assert_matches(scores, threshold, cooldown, gs, ge, receipts)


def test_handcrafted_backfill_and_pending():
    # 阈值=5，冷却=2，缺报覆盖全部帧且无回执 -> 三种 r 全部一致
    scores = [8, 8, 7, 8, 7, 4, 3, 5, 6, 3, 8, 9]
    got = assert_matches(scores, 5, 2, 0, 11, [])
    # 手工追踪（触发 -> 冷却两帧）：
    # r=0: [0, 3, 7, 10]；r=1: [1, 4, 7, 10]；r=2: [2, 7, 10]
    assert brute_triggers(scores, 5, 2, 0) == [0, 3, 7, 10]
    assert brute_triggers(scores, 5, 2, 1) == [1, 4, 7, 10]
    assert brute_triggers(scores, 5, 2, 2) == [2, 7, 10]
    # 交集 = 补记 {7,10}；并集减交集 = 待核 {0,1,2,3,4}
    assert got["status"] == "ok"
    assert got["consistent_rs"] == [0, 1, 2]
    assert got["backfill"] == [7, 10]
    assert got["pending"] == [0, 1, 2, 3, 4]


def test_cooldown_zero_chain():
    # 冷却 0：每个达阈帧都触发，链最长，检验迭代式记忆化不爆栈且结果正确
    scores = [1, 0, 1, 1, 0, 1] * 100
    # 缺报覆盖全片、无区间外回执：唯一解释 r=0 下所有达阈帧都进补记
    got = assert_matches(scores, 1, 0, 0, len(scores) - 1, [])
    assert got["status"] == "ok"
    # 冷却 0 只有 r=0 一种解释；区间内所有达阈帧必然全在补记里，无待核
    assert got["pending"] == []
    assert got["backfill"] == [i for i in range(len(scores)) if scores[i] >= 1]


def test_full_gap_and_no_receipts():
    rng = random.Random(5)
    scores = [rng.randint(0, 5) for _ in range(60)]
    got = assert_matches(scores, 3, 4, 0, 59, [])
    # 无回执 + 全区间缺报：所有 r 都一致
    assert got["status"] == "ok"
    assert got["consistent_rs"] == list(range(5))


def test_conflict_when_receipt_never_fires():
    # 回执帧在任何 r 下都不达标 -> 冲突，补记留空
    scores = [0, 0, 0, 0, 0]
    res = verify(scores, 1, 2, 0, 4, [2])
    assert res.status == "conflict"
    assert res.backfill == []
    assert res.pending == []
    assert len(res.rejections) == 3  # r = 0,1,2 全部被否决


def test_conflict_extra_trigger_outside_gap():
    # 区间外必然触发的帧不在回执里 -> 冲突
    scores = [9, 9]
    res = verify(scores, 5, 0, 1, 1, [])
    assert res.status == "conflict"
    assert all("帧 0" in row["reason"] for row in res.rejections)


def test_receipt_inside_gap_still_must_fire():
    # 区间内回执帧不达标 -> 仍算冲突（缺报不豁免已回执帧的真实性）
    scores = [0, 0]
    res = verify(scores, 1, 0, 0, 1, [1])
    assert res.status == "conflict"


def test_received_frames_in_gap_marked():
    scores = [5, 5, 5]
    res = verify(scores, 5, 0, 0, 2, [1])
    assert res.status == "ok"
    by_frame = {row["frame"]: row["status"] for row in res.backfill}
    assert by_frame == {0: "backfill", 1: "received", 2: "backfill"}


def test_input_validation():
    with pytest.raises(VerifyError):
        verify([], 1, 0, 0, 0, [])
    with pytest.raises(VerifyError):
        verify([1], 1, MAX_COOLDOWN + 1, 0, 0, [])
    with pytest.raises(VerifyError):
        verify([1, 2], 1, 0, 0, 2, [])  # 终点越界
    with pytest.raises(VerifyError):
        verify([1, 2], 1, 0, 1, 0, [])  # 起点 > 终点
    with pytest.raises(VerifyError):
        verify([1, 2], 1, 0, 0, 1, [0, 0])  # 回执重复
    with pytest.raises(VerifyError):
        verify([1, 2], 1, 0, 0, 1, [5])  # 回执越界
    with pytest.raises(VerifyError):
        verify([1, 2.0], 1, 0, 0, 1, [])  # 非整数分数


def test_max_size_performance():
    # 20 万帧 + 最大冷却：必须一次 O(N) 级完成，不能随 r 数量逐帧重扫
    rng = random.Random(1)
    n = MAX_FRAMES
    scores = [rng.randint(0, 100) for _ in range(n)]
    import time

    t0 = time.perf_counter()
    res = verify(scores, 50, MAX_COOLDOWN, 1000, n - 2, [])
    elapsed = time.perf_counter() - t0
    assert res.status == "ok"
    # 记忆化状态数远小于 (cooldown+1)*n，证明没有逐 r 重扫
    assert res.stats["memoized_states"] <= n + MAX_COOLDOWN + 2
    assert elapsed < 5.0
