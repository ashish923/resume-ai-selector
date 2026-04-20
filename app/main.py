"""
ResumeAI Selector — Main Application
"""

import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import settings
from app.routes import upload, dashboard, analysis

# Create required directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title=settings.APP_TITLE)

# Static files & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Register routes
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": settings.APP_TITLE,
    })
