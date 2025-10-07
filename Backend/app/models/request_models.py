from typing import List, Optional
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    search_id: str
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=10, description="User rating from 1 (poor) to 10 (excellent)")
    comment: Optional[str] = None
