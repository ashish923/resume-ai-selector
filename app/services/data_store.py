"""
In-memory data store.
Holds parsed resumes and job description during the session.
"""

from typing import Optional


class DataStore:
    """Simple in-memory store for resume data and job description"""

    def __init__(self):
        self.resumes: dict = {}        # filename -> parsed resume dict
        self.job_description: str = "" # current JD text
        self.results: list = []        # scored/ranked results

    def add_resume(self, resume_data: dict):
        """Add a parsed resume to the store"""
        filename = resume_data["filename"]
        self.resumes[filename] = resume_data

    def get_resume(self, filename: str) -> Optional[dict]:
        """Get a specific resume by filename"""
        return self.resumes.get(filename)

    def get_all_resumes(self) -> list:
        """Get all stored resumes as a list"""
        return list(self.resumes.values())

    def get_resume_count(self) -> int:
        """Get total number of resumes"""
        return len(self.resumes)

    def remove_resume(self, filename: str):
        """Remove a resume from the store"""
        self.resumes.pop(filename, None)

    def clear_resumes(self):
        """Clear all resumes"""
        self.resumes.clear()

    def set_job_description(self, jd_text: str):
        """Set the current job description"""
        self.job_description = jd_text

    def get_job_description(self) -> str:
        """Get the current job description"""
        return self.job_description

    def set_results(self, results: list):
        """Store scored/ranked results"""
        self.results = results

    def get_results(self) -> list:
        """Get scored/ranked results"""
        return self.results

    def clear_all(self):
        """Reset everything"""
        self.resumes.clear()
        self.job_description = ""
        self.results.clear()


# Single global instance — shared across the app
store = DataStore()