from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TextCorpus(BaseModel):
    texts: List[str]
    document_ids: Optional[List[str]] = None

class SingleText(BaseModel):
    text: str

class LSARequest(BaseModel):
    texts: List[str]
    n_components: int = 5

class ProcessingResponse(BaseModel):
    status: str
    data: Any
    message: Optional[str] = None
    execution_time: Optional[float] = None

class NLTKResponse(BaseModel):
    original_text: str
    processed_data: Any
    message: Optional[str] = None