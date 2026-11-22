"""FastAPI 入口：核对接口、补记 CSV 下载、静态页面。

每次重新核对都会使上一次的结论和下载链接失效（旧链接返回 410），
前端也会在重新核对时清空旧结论与下载入口。
"""

from __future__ import annotations

import csv
import io
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import MAX_COOLDOWN, MAX_FRAMES, VerifyError, verify

app = FastAPI(title="唤醒缺报核对", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"


class CheckRequest(BaseModel):
    scores: list[int] = Field(..., description="逐帧模型整数分数，帧号从 0 起")
    threshold: int
    cooldown: int
    gap_start: int
    gap_end: int
    receipts: list[int] = Field(default_factory=list, description="已收到的触发帧号")


class ResultStore:
    """只保留最近一次核对结果；重新核对即作废旧结论与旧下载。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_id: str | None = None
        self._results: dict[str, dict] = {}

    def put(self, result: dict) -> str:
        with self._lock:
            self._results.clear()  # 作废旧结论与旧下载链接
            rid = uuid.uuid4().hex
            self._results[rid] = result
            self._current_id = rid
            return rid

    def get(self, rid: str) -> dict | None:
        with self._lock:
            return self._results.get(rid)


store = ResultStore()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/limits")
def limits() -> dict:
    return {"max_frames": MAX_FRAMES, "max_cooldown": MAX_COOLDOWN}


@app.post("/api/check")
def check(req: CheckRequest) -> dict:
    try:
        result = verify(
            scores=req.scores,
            threshold=req.threshold,
            cooldown=req.cooldown,
            gap_start=req.gap_start,
            gap_end=req.gap_end,
            receipts=req.receipts,
        )
    except VerifyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = result.__dict__
    rid = store.put(payload)
    return {"result_id": rid, **payload}


@app.get("/api/results/{rid}/backfill.csv")
def download_backfill(rid: str) -> StreamingResponse:
    result = store.get(rid)
    if result is None:
        # 链接曾经存在但已被新一轮核对撤下，或根本不存在
        raise HTTPException(status_code=410, detail="该补记下载已失效：请重新核对。")
    if result["status"] != "ok":
        raise HTTPException(status_code=409, detail="冲突状态下没有可补记列表。")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["frame", "score", "status"])
    for row in result["backfill"]:
        writer.writerow([row["frame"], row["score"], row["status"]])
    data = buf.getvalue().encode("utf-8-sig")

    filename = f"backfill_{rid[:8]}.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
