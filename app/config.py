import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Hugging Face
    HF_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "data/chroma_db")

    # App
    APP_TITLE: str = "ResumeAI Selector"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx"}


settings = Settings()
