from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    namespace: str = Field(default="default", min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    score: float | None = None
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
