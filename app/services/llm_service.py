"""
Service: LLM Integration
Connects to Hugging Face Inference Providers.
"""

from huggingface_hub import InferenceClient
from app.config import settings

# Module-level cached instance
_client = None


def get_client():
    """Initialize and return the HuggingFace InferenceClient."""
    global _client
    if _client is None:
        if not settings.HF_API_TOKEN:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN is not set. "
                "Add it to your .env file."
            )

        _client = InferenceClient(
            provider="auto",
            api_key=settings.HF_API_TOKEN,
        )
    return _client


def ask_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """
    Send a system + user prompt to the LLM and get raw text response.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )

    return response.choices[0].message.content


def test_connection() -> dict:
    """Test if the LLM connection is working"""
    try:
        result = ask_llm(
            system_prompt="You are a helpful assistant. Reply briefly.",
            user_prompt="Say 'Hello, ResumeAI is connected!' and nothing else.",
            max_tokens=50,
        )
        return {
            "status": "connected",
            "model": settings.LLM_MODEL,
            "response": result.strip(),
        }
    except Exception as e:
        return {
            "status": "error",
            "model": settings.LLM_MODEL,
            "error": str(e),
        }


def extract_resume_info(resume_text: str) -> dict:
    """
    Use LLM to extract structured information from resume text.
    Returns dict with: name, email, phone, skills, experience_years,
                       job_title, education, summary
    """
    import json

    system_prompt = """You are a resume parser. Extract structured information from the given resume text.
Return ONLY a valid JSON object with these exact keys:
{
    "name": "Full name of the candidate",
    "email": "Email address or empty string",
    "phone": "Phone number or empty string",
    "job_title": "Most recent job title",
    "skills": ["skill1", "skill2", "skill3"],
    "experience_years": 0,
    "education": "Highest degree and institution",
    "summary": "2-3 sentence professional summary"
}

Rules:
- skills should be a flat list of technical and professional skills
- experience_years should be a number (integer)
- If information is not found, use empty string or empty list
- Return ONLY the JSON, no extra text, no markdown, no code blocks"""

    user_prompt = f"Extract information from this resume:\n\n{resume_text[:3000]}"

    try:
        response = ask_llm(system_prompt, user_prompt, max_tokens=800)

        # Clean the response — remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        # Parse JSON
        result = json.loads(cleaned)

        # Ensure all required keys exist
        defaults = {
            "name": "", "email": "", "phone": "",
            "job_title": "", "skills": [],
            "experience_years": 0, "education": "",
            "summary": ""
        }
        for key, default in defaults.items():
            if key not in result:
                result[key] = default

        return result

    except json.JSONDecodeError:
        return {
            "name": "Unknown",
            "email": "",
            "phone": "",
            "job_title": "Unknown",
            "skills": [],
            "experience_years": 0,
            "education": "",
            "summary": "Failed to parse resume automatically.",
            "raw_response": response,
        }
    except Exception as e:
        return {
            "name": "Error",
            "skills": [],
            "experience_years": 0,
            "summary": f"Extraction failed: {str(e)}",
        }


def extract_jd_requirements(jd_text: str) -> dict:
    """
    Use LLM to extract structured requirements from a job description.
    Returns dict with: title, required_skills, preferred_skills,
                       min_experience, education, responsibilities
    """
    import json

    system_prompt = """You are a job description parser. Extract structured requirements from the given job description.
Return ONLY a valid JSON object with these exact keys:
{
    "title": "Job title",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "min_experience_years": 0,
    "education": "Required education level",
    "responsibilities": ["responsibility1", "responsibility2"]
}

Rules:
- required_skills: skills that are mandatory (must-have)
- preferred_skills: skills that are nice-to-have or bonus
- min_experience_years: minimum years of experience required (integer)
- If not specified, set min_experience_years to 0
- Return ONLY the JSON, no extra text, no markdown, no code blocks"""

    user_prompt = f"Extract requirements from this job description:\n\n{jd_text[:3000]}"

    try:
        response = ask_llm(system_prompt, user_prompt, max_tokens=800)

        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        result = json.loads(cleaned)

        # Ensure all required keys exist
        defaults = {
            "title": "", "required_skills": [],
            "preferred_skills": [], "min_experience_years": 0,
            "education": "", "responsibilities": []
        }
        for key, default in defaults.items():
            if key not in result:
                result[key] = default

        return result

    except json.JSONDecodeError:
        return {
            "title": "Unknown",
            "required_skills": [],
            "preferred_skills": [],
            "min_experience_years": 0,
            "education": "",
            "responsibilities": [],
            "raw_response": response,
        }
    except Exception as e:
        return {
            "title": "Error",
            "required_skills": [],
            "preferred_skills": [],
            "min_experience_years": 0,
            "responsibilities": [],
            "summary": f"Extraction failed: {str(e)}",
        }