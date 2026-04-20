"""
Routes: Dashboard
Displays ranked candidates and comparison views.
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

from app.services.data_store import store

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def dashboard_page(request: Request):
    """Render the dashboard with ranked candidates"""
    results = store.get_results()
    jd_text = store.get_job_description()

    # Calculate stats
    total = len(results)
    strong = len([r for r in results if r["score"]["match_level"] == "strong"])
    moderate = len([r for r in results if r["score"]["match_level"] == "moderate"])
    weak = len([r for r in results if r["score"]["match_level"] == "weak"])

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Candidate Dashboard",
        "candidates": results,
        "total": total,
        "strong": strong,
        "moderate": moderate,
        "weak": weak,
        "has_results": total > 0,
    })