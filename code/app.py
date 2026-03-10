"""
FastAPI Application Entry Point
================================
นี่คือไฟล์หลักที่เปิดตัว web server

สิ่งที่ไฟล์นี้ทำ:
1. สร้าง FastAPI app object
2. เชื่อมต่อ router (เส้นทาง API ทั้งหมดอยู่ใน router/route_v1.py)
3. เสิร์ฟไฟล์ static (รูป, CSS, JS)
4. เสิร์ฟหน้าเว็บ HTML หลัก (static/index.html)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from router.route_v1 import api_v1

app = FastAPI(
    title="Restbiz — น้องสุดยอด",
    description="Thai Regulatory AI Assistant for restaurant businesses",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1, prefix="/api/v1")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>index.html not found in code/static/</h1>", status_code=404)

    html = HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    html.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    html.headers["Pragma"] = "no-cache"
    html.headers["Expires"] = "0"
    return html