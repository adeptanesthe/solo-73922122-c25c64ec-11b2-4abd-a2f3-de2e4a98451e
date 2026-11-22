"""API 层测试：核对、CSV 下载、重新核对后旧链接失效（旧结论/旧下载撤下）。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post_check(receipts=None, **over):
    payload = {
        "scores": [5, 5, 0, 5, 5, 5],
        "threshold": 5,
        "cooldown": 1,
        "gap_start": 2,
        "gap_end": 4,
        # r=0 触发 [0,3,5]、r=1 触发 [1,3,5]；区间外回执 {0,5} 使 r=0 一致
        "receipts": receipts if receipts is not None else [0, 5],
    }
    payload.update(over)
    return client.post("/api/check", json=payload)


def test_health_and_limits():
    assert client.get("/api/health").json() == {"status": "ok"}
    limits = client.get("/api/limits").json()
    assert limits["max_frames"] == 200_000
    assert limits["max_cooldown"] == 2000


def test_check_ok_and_csv():
    resp = post_check()
    assert resp.status_code == 200
    data = resp.json()
    rid = data["result_id"]
    assert data["status"] == "ok"
    assert data["backfill"], "缺报区间内应有可补记帧"

    csv_resp = client.get(f"/api/results/{rid}/backfill.csv")
    assert csv_resp.status_code == 200
    text = csv_resp.content.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0] == "frame,score,status"
    assert len(lines) == 1 + len(data["backfill"])
    assert f'attachment; filename="backfill_{rid[:8]}.csv"' in csv_resp.headers["content-disposition"]


def test_conflict_has_no_csv_and_lists_rejections():
    # 回执帧 2 在区间内但永不达标（该帧分数为 0）
    resp = post_check(receipts=[2])
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "conflict"
    assert data["backfill"] == []
    assert len(data["rejections"]) == 2  # r = 0, 1
    csv_resp = client.get(f"/api/results/{data['result_id']}/backfill.csv")
    assert csv_resp.status_code == 409


def test_recheck_invalidates_old_result_and_download():
    first = post_check().json()
    old_id = first["result_id"]
    assert client.get(f"/api/results/{old_id}/backfill.csv").status_code == 200

    # 更正缺报区间后重新核对
    second = post_check(gap_start=1, gap_end=3).json()
    assert second["result_id"] != old_id

    # 旧结论数据与旧下载都必须撤下
    assert client.get(f"/api/results/{old_id}/backfill.csv").status_code == 410
    assert client.get(f"/api/results/{second['result_id']}/backfill.csv").status_code == 200


def test_bad_input_returns_400():
    resp = post_check(cooldown=9999)
    assert resp.status_code == 400
    assert "冷却帧数" in resp.json()["detail"]


def test_index_page_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "唤醒词缺报核对" in resp.text
