from typing import List, Optional
from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
