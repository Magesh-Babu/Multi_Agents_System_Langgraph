from pydantic import BaseModel
from typing import List


class AnalyzeRequest(BaseModel):
    query: str
    thread_id: str = "default"


class AnalyzeResponse(BaseModel):
    answer: str
    agents_used: List[str]
    thread_id: str
