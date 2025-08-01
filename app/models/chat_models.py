# backend/app/models/chat_models.py

from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from app.core.config import settings
# In app/models/chat_models.py
from pydantic import BaseModel, HttpUrl
# =================================================================
#  Models for Chat, Documents, and RAG
# =================================================================

class Message(BaseModel):
    role: str # "user", "assistant", or "system"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None
    document_ids: Optional[List[str]] = None
    llm_provider: Optional[str] = settings.DEFAULT_LLM_PROVIDER
    language: Optional[str] = "auto"

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str


class URLIngestRequest(BaseModel):
    """Request model for ingesting content from a URL."""
    url: HttpUrl

# =================================================================
#  Models for User Authentication
# =================================================================

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[EmailStr] = None