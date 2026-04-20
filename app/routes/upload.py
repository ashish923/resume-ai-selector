"""
Routes: Upload
Handles resume file uploads and JD input.
"""

import os
import uuid
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from typing import List

from app.config import settings
from app.services.resume_parser import parse_resume
from app.services.data_store import store

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def upload_page(request: Request):
    """Render the upload page"""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": "Upload Resumes",
        "resume_count": store.get_resume_count(),
        "resumes": store.get_all_resumes(),
    })


@router.post("/resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """
    Handle bulk resume upload.
    - Validates file type and size
    - Saves to uploads/ folder
    - Parses text from each file
    - Stores parsed data in memory
    """
    uploaded = []
    errors = []

    for file in files:
        # Get file extension
        ext = os.path.splitext(file.filename)[1].lower()

        # Validate extension
        if ext not in settings.ALLOWED_EXTENSIONS:
            errors.append(f"{file.filename}: Unsupported file type '{ext}'. Only PDF and DOCX allowed.")
            continue

        # Read file content
        content = await file.read()

        # Validate file size
        if len(content) > settings.MAX_FILE_SIZE:
            errors.append(f"{file.filename}: File too large. Max {settings.MAX_FILE_SIZE // (1024*1024)}MB.")
            continue

        # Generate unique filename to avoid conflicts
        unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_name)

        # Save file to disk
        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            errors.append(f"{file.filename}: Failed to save — {str(e)}")
            continue

        # Parse the resume and extract text
        try:
            resume_data = parse_resume(file_path)
            resume_data["original_filename"] = file.filename
            store.add_resume(resume_data)
            uploaded.append({
                "filename": file.filename,
                "text_length": resume_data["text_length"],
                "preview": resume_data["raw_text"][:200] + "...",
            })
        except Exception as e:
            errors.append(f"{file.filename}: Parse error — {str(e)}")
            # Clean up saved file if parsing failed
            if os.path.exists(file_path):
                os.remove(file_path)

    return JSONResponse(content={
        "uploaded": uploaded,
        "errors": errors,
        "total_stored": store.get_resume_count(),
    })


@router.post("/job-description")
async def submit_jd(jd_text: str = Form(...)):
    """Save the job description text"""
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    store.set_job_description(jd_text.strip())

    return JSONResponse(content={
        "message": "Job description saved",
        "length": len(jd_text.strip()),
    })


@router.delete("/resumes/{filename}")
async def delete_resume(filename: str):
    """Remove a resume from the store"""
    resume = store.get_resume(filename)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    # Delete file from disk
    if os.path.exists(resume["file_path"]):
        os.remove(resume["file_path"])

    store.remove_resume(filename)

    return JSONResponse(content={
        "message": f"Removed {filename}",
        "total_stored": store.get_resume_count(),
    })


@router.delete("/clear")
async def clear_all():
    """Clear all resumes and JD"""
    # Delete all uploaded files
    for resume in store.get_all_resumes():
        if os.path.exists(resume["file_path"]):
            os.remove(resume["file_path"])

    store.clear_all()

    return JSONResponse(content={"message": "All data cleared"})



@router.get("/test-llm")
async def test_llm():
    """Quick endpoint to test if HuggingFace LLM is connected"""
    from app.services.llm_service import test_connection
    return test_connection()


@router.get("/test-extract")
async def test_extract():
    """Test resume extraction on the first uploaded resume"""
    from app.services.llm_service import extract_resume_info

    resumes = store.get_all_resumes()
    if not resumes:
        return {"error": "No resumes uploaded. Upload one first at /upload"}

    first_resume = resumes[0]
    result = extract_resume_info(first_resume["raw_text"])

    return {
        "filename": first_resume["filename"],
        "extracted": result,
    }

@router.get("/test-jd")
async def test_jd():
    """Test JD extraction on the saved job description"""
    from app.services.llm_service import extract_jd_requirements

    jd_text = store.get_job_description()
    if not jd_text:
        return {"error": "No job description saved. Submit one at /upload first."}

    result = extract_jd_requirements(jd_text)

    return {"extracted": result}

@router.post("/analyze")
async def analyze_resumes():
    """
    Run the scoring engine:
    1. Extract JD requirements
    2. Extract info from each resume
    3. Score and rank all resumes
    """
    from app.services.scorer import rank_resumes

    # Check we have data
    resumes = store.get_all_resumes()
    jd_text = store.get_job_description()

    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes uploaded.")
    if not jd_text:
        raise HTTPException(status_code=400, detail="No job description saved.")

    # Run scoring
    results, jd_info = rank_resumes(resumes, jd_text)

    # Store results for dashboard
    store.set_results(results)

    return JSONResponse(content={
        "message": f"Scored {len(results)} resumes",
        "jd_title": jd_info.get("title", ""),
        "results": results,
    })