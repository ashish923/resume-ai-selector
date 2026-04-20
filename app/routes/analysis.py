"""
Routes: Analysis
RAG-based chat with resume pool (with JD context + memory).
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from app.services.data_store import store
from app.services.vector_store import add_resumes_to_store, query_resumes

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# In-memory chat history (resets on server restart)
chat_history: List[dict] = []


class ChatRequest(BaseModel):
    question: str


@router.get("/")
async def analysis_page(request: Request):
    """Render the chat page"""
    resumes = store.get_all_resumes()
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "title": "Chat with Resumes",
        "resume_count": len(resumes),
        "resumes": resumes,
    })


@router.post("/index")
async def index_resumes():
    """Index all uploaded resumes into ChromaDB."""
    resumes = store.get_all_resumes()

    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes uploaded.")

    result = add_resumes_to_store(resumes)

    # Clear chat history on re-index
    chat_history.clear()

    return JSONResponse(content={
        "message": "Resumes indexed successfully",
        "details": result,
    })


@router.post("/chat")
async def chat_with_resumes(req: ChatRequest):
    """
    RAG query with JD context and chat memory.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    question = req.question.strip()

    # Get JD text for context
    jd_text = store.get_job_description()

    # Query with JD + history
    answer = query_resumes(
        question=question,
        jd_text=jd_text,
        chat_history=chat_history,
    )

    # Store in chat history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return JSONResponse(content={
        "question": question,
        "answer": answer,
    })


@router.delete("/clear-chat")
async def clear_chat():
    """Clear chat history"""
    chat_history.clear()
    return JSONResponse(content={"message": "Chat history cleared"})