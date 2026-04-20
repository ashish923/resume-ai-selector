"""
Service: Resume Scorer (Hybrid — Keyword + LLM)
Scores and ranks resumes against a job description.
Uses keyword matching first, then LLM for unmatched skills.
"""

import json
from app.services.llm_service import extract_resume_info, extract_jd_requirements, ask_llm


def keyword_match(resume_skills: list, jd_skills: list) -> dict:
    """
    Step 1: Fast keyword matching (case-insensitive, partial match).
    Returns matched, unmatched_resume, unmatched_jd lists.
    """
    if not jd_skills:
        return {"matched": [], "unmatched_jd": [], "unmatched_resume": resume_skills}

    resume_lower = [s.lower().strip() for s in resume_skills]
    jd_lower = [s.lower().strip() for s in jd_skills]

    matched_jd = []
    unmatched_jd = []

    for skill in jd_skills:
        skill_lower = skill.lower().strip()
        found = False
        for r_skill in resume_lower:
            if skill_lower in r_skill or r_skill in skill_lower:
                found = True
                break
        if found:
            matched_jd.append(skill)
        else:
            unmatched_jd.append(skill)

    # Find resume skills not matched to any JD skill
    matched_lower = [s.lower() for s in matched_jd]
    unmatched_resume = [s for s in resume_skills
                        if not any(m in s.lower() or s.lower() in m for m in matched_lower)]

    return {
        "matched": matched_jd,
        "unmatched_jd": unmatched_jd,
        "unmatched_resume": unmatched_resume,
    }


def llm_skill_match(unmatched_jd: list, unmatched_resume: list) -> list:
    """
    Step 2: Use LLM to find semantic matches between
    unmatched JD skills and unmatched resume skills.
    Only called when keyword matching leaves gaps.
    """
    if not unmatched_jd or not unmatched_resume:
        return []

    system_prompt = """You are a skill matching expert. 
Given a list of JD required skills and a candidate's skills, 
find which candidate skills are semantically equivalent to JD skills.

Return ONLY a valid JSON array of matches like:
[
    {"jd_skill": "Machine Learning", "resume_skill": "ML", "reason": "abbreviation"},
    {"jd_skill": "LLMs", "resume_skill": "GenAI", "reason": "related field"}
]

Rules:
- Only match skills that are genuinely related
- Don't force matches — if no match exists, return empty array []
- Return ONLY the JSON array, no extra text"""

    user_prompt = f"""JD skills (unmatched): {unmatched_jd}
Candidate skills (unmatched): {unmatched_resume}

Find semantic matches:"""

    try:
        response = ask_llm(system_prompt, user_prompt, max_tokens=400)

        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        matches = json.loads(cleaned)
        if isinstance(matches, list):
            return matches
        return []

    except (json.JSONDecodeError, Exception):
        return []


def hybrid_skill_match(resume_skills: list, jd_skills: list) -> dict:
    """
    Hybrid matching: Keyword first, then LLM for remaining.
    Returns matched skills, missing skills, and match percentage.
    """
    if not jd_skills:
        return {"matched": [], "missing": [], "score": 100.0, "method": "no_requirements"}

    # Step 1: Keyword matching
    kw_result = keyword_match(resume_skills, jd_skills)

    keyword_matched = kw_result["matched"]
    unmatched_jd = kw_result["unmatched_jd"]
    unmatched_resume = kw_result["unmatched_resume"]

    # Step 2: LLM matching (only for unmatched skills)
    llm_matched_jd = []
    if unmatched_jd and unmatched_resume:
        llm_matches = llm_skill_match(unmatched_jd, unmatched_resume)
        for match in llm_matches:
            jd_skill = match.get("jd_skill", "")
            if jd_skill in unmatched_jd:
                llm_matched_jd.append(jd_skill)
                unmatched_jd.remove(jd_skill)

    # Combine results
    all_matched = keyword_matched + llm_matched_jd
    missing = unmatched_jd

    score = (len(all_matched) / len(jd_skills)) * 100 if jd_skills else 100

    return {
        "matched": all_matched,
        "keyword_matched": keyword_matched,
        "llm_matched": llm_matched_jd,
        "missing": missing,
        "score": round(score, 1),
    }


def calculate_experience_score(resume_years: int, jd_min_years: int) -> dict:
    """Score experience match."""
    if jd_min_years == 0:
        return {"score": 100.0, "resume_years": resume_years, "required_years": 0}

    if resume_years >= jd_min_years:
        score = 100.0
    else:
        score = (resume_years / jd_min_years) * 100

    return {
        "score": round(score, 1),
        "resume_years": resume_years,
        "required_years": jd_min_years,
    }


def calculate_education_score(resume_education: str, jd_education: str) -> dict:
    """Simple education match using keyword overlap."""
    if not jd_education or not resume_education:
        return {"score": 100.0, "matched": True}

    resume_lower = resume_education.lower()
    jd_lower = jd_education.lower()

    keywords = ["phd", "master", "bachelor", "b.tech", "b.e", "m.tech",
                "m.e", "mba", "bca", "mca", "b.sc", "m.sc", "diploma",
                "computer science", "engineering", "information technology"]

    jd_keywords = [k for k in keywords if k in jd_lower]

    if not jd_keywords:
        return {"score": 100.0, "matched": True}

    matched = any(k in resume_lower for k in jd_keywords)
    return {"score": 100.0 if matched else 30.0, "matched": matched}


def score_resume(resume_info: dict, jd_info: dict) -> dict:
    """
    Score a single resume against JD using hybrid matching.

    Weights:
    - Required skills: 50% (hybrid: keyword + LLM)
    - Preferred skills: 20% (hybrid: keyword + LLM)
    - Experience: 20%
    - Education: 10%
    """
    required_match = hybrid_skill_match(
        resume_info.get("skills", []),
        jd_info.get("required_skills", [])
    )

    preferred_match = hybrid_skill_match(
        resume_info.get("skills", []),
        jd_info.get("preferred_skills", [])
    )

    experience = calculate_experience_score(
        resume_info.get("experience_years", 0),
        jd_info.get("min_experience_years", 0)
    )

    education = calculate_education_score(
        resume_info.get("education", ""),
        jd_info.get("education", "")
    )

    final_score = (
        required_match["score"] * 0.50 +
        preferred_match["score"] * 0.20 +
        experience["score"] * 0.20 +
        education["score"] * 0.10
    )

    if final_score >= 75:
        match_level = "strong"
    elif final_score >= 50:
        match_level = "moderate"
    else:
        match_level = "weak"

    return {
        "final_score": round(final_score, 1),
        "match_level": match_level,
        "required_skills": required_match,
        "preferred_skills": preferred_match,
        "experience": experience,
        "education": education,
    }


def rank_resumes(resumes: list, jd_text: str) -> list:
    """
    Score and rank ALL resumes against the JD.
    Uses hybrid scoring (keyword + LLM).
    """
    jd_info = extract_jd_requirements(jd_text)

    results = []

    for resume in resumes:
        resume_info = extract_resume_info(resume["raw_text"])
        score_data = score_resume(resume_info, jd_info)

        results.append({
            "filename": resume.get("original_filename", resume["filename"]),
            "candidate_name": resume_info.get("name", "Unknown"),
            "job_title": resume_info.get("job_title", ""),
            "email": resume_info.get("email", ""),
            "phone": resume_info.get("phone", ""),
            "skills": resume_info.get("skills", []),
            "experience_years": resume_info.get("experience_years", 0),
            "education": resume_info.get("education", ""),
            "summary": resume_info.get("summary", ""),
            "score": score_data,
        })

    results.sort(key=lambda x: x["score"]["final_score"], reverse=True)

    return results, jd_info