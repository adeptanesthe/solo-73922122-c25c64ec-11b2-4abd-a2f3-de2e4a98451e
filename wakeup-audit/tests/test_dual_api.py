"""双音箱试跑 API 测试：建会话、分块追加、整块退回、结束与拒收。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_session(**over):
    payload = {
        "threshold_1": 80,
        "cooldown_1": 2,
        "threshold_2": 70,
        "cooldown_2": 3,
        "w": 2,
    }
    payload.update(over)
    return client.post("/api/dual/sessions", json=payload)


def test_limits_include_dual_constants():
    data = client.get("/api/limits").json()
    assert data["dual"] == {
        "max_frames": 200_000,
        "max_cooldown": 2000,
        "max_w": 50,
        "max_lead": 4096,
    }


def test_create_and_append_full_flow():
    sid = create_session().json()["session_id"]

    # 首块必须从 0 开始：故意写错起始帧 -> 400 且整块退回
    bad = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 1, "start_frame": 1, "block": [90]},
    )
    assert bad.status_code == 400
    assert "起始帧应为 0" in bad.json()["detail"]

    # 两路各 6 帧：一号帧 0 达 80（90），二号帧 0 也达 70（75），窗 [0..2]
    r1 = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 1, "start_frame": 0, "block": [90, 0, 0, 0, 0, 85]},
    )
    assert r1.status_code == 200
    r2 = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 2, "start_frame": 0, "block": [75, 0, 0, 0, 95, 0]},
    )
    assert r2.status_code == 200
    snap = r2.json()
    assert snap["lengths"] == [6, 6]
    assert snap["frontier"] == 6  # 共同帧全部处理完；第一轮在处理帧 2 时即已确定
    # 第一轮：一号 90 分胜，在窗末帧 2 应答
    assert len(snap["answers"]) == 1
    a = snap["answers"][0]
    assert a["speaker"] == 1 and a["selected_frame"] == 0 and a["answer_frame"] == 2
    assert a["window_start"] == 0 and a["window_end"] == 2
    # 胜者冷却快照：一号 avail=5，frontier=6 -> 剩余 0；二号未冷却
    assert snap["cooldown_remaining"] == [0, 0]

    # 等长结束：帧 4 二号 95 开窗（一号仍在冷却：avail=2+2+1=5，帧 4 冷却），
    # 末窗 [4..5] 按最后一帧 5 结算，二号胜
    fin = client.post(f"/api/dual/sessions/{sid}/finish")
    assert fin.status_code == 200
    data = fin.json()
    assert data["finished"] is True
    assert len(data["answers"]) == 2
    b = data["answers"][1]
    assert b["speaker"] == 2 and b["selected_frame"] == 4 and b["answer_frame"] == 5

    # 成功结束后拒收新块
    again = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 1, "start_frame": 6, "block": [1]},
    )
    assert again.status_code == 400
    assert "回放已结束" in again.json()["detail"]


def test_unequal_lengths_finish_rejected_then_resume():
    sid = create_session(w=0).json()["session_id"]
    client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 1, "start_frame": 0, "block": [90, 0, 0]},
    )
    client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 2, "start_frame": 0, "block": [0, 0]},
    )
    fin = client.post(f"/api/dual/sessions/{sid}/finish")
    assert fin.status_code == 400
    assert "两路帧数不一致" in fin.json()["detail"]

    # 会话仍可用：补齐二号后成功结束
    ok = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 2, "start_frame": 2, "block": [0]},
    )
    assert ok.status_code == 200
    fin2 = client.post(f"/api/dual/sessions/{sid}/finish")
    assert fin2.status_code == 200
    assert fin2.json()["finished"] is True


def test_lead_limit_rejected_atomically():
    sid = create_session().json()["session_id"]
    # 二号空，一号追 4097 帧 -> 超过 4096，整块退回
    resp = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 1, "start_frame": 0, "block": [0] * 4097},
    )
    assert resp.status_code == 400
    assert "最大领先" in resp.json()["detail"]
    status = client.get(f"/api/dual/sessions/{sid}").json()
    assert status["lengths"] == [0, 0]
    assert status["blocks"] == []


def test_each_append_reports_determined_answers_only():
    sid = create_session(w=50).json()["session_id"]
    # 一号 4 帧、二号 3 帧（不等长），帧 0 即达标但窗末未到 -> 不结算
    client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 1, "start_frame": 0, "block": [90, 0, 0, 0]},
    )
    r = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 2, "start_frame": 0, "block": [0, 0, 0]},
    )
    assert r.json()["answers"] == []
    # 补齐等长并结束：末窗按最后一帧（帧 3）结算
    client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 2, "start_frame": 3, "block": [0]},
    )
    fin = client.post(f"/api/dual/sessions/{sid}/finish").json()
    assert fin["answers"][0]["answer_frame"] == 3


def test_bad_config_and_unknown_session():
    assert create_session(cooldown_1=2001).status_code == 400
    assert create_session(w=51).status_code == 400
    assert client.get("/api/dual/sessions/nope").status_code == 404
    assert client.post("/api/dual/sessions/nope/finish").status_code == 404


def test_non_integer_block_rejected():
    sid = create_session().json()["session_id"]
    resp = client.post(
        f"/api/dual/sessions/{sid}/blocks",
        json={"channel": 1, "start_frame": 0, "block": [1, 2.5]},
    )
    # pydantic 层（422）或引擎层（400）都算拒绝，且必须没落块
    assert resp.status_code in (400, 422)
    assert client.get(f"/api/dual/sessions/{sid}").json()["lengths"] == [0, 0]
