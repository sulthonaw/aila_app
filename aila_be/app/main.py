import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.core.models import ChatRequest, ChatResponse, SourceChunk
from app.services.rag_service import RagService

rag_service: RagService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag_service
    rag_service = RagService()
    yield


app = FastAPI(title="RAG Backend", version="1.0.0", lifespan=lifespan)


def _build_chat_response(result: dict) -> ChatResponse:
    sources = [
        SourceChunk(
            source=item["source"],
            page=item.get("page"),
            score=item.get("score"),
            content=item["content"],
        )
        for item in result["sources"]
    ]
    return ChatResponse(answer=result["answer"], sources=sources)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    namespace: str = Form("default"),
) -> dict:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        result = rag_service.ingest_pdf(
            file_path=tmp_path,
            source_name=file.filename,
            namespace=namespace,
        )
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    result = rag_service.chat(
        question=req.question,
        namespace=req.namespace,
        top_k=req.top_k,
    )

    return _build_chat_response(result)


@app.post("/chat/attach", response_model=ChatResponse)
async def chat_with_attachment(
    question: str = Form(...),
    file: UploadFile = File(...),
    namespace: str = Form("default"),
    top_k: int = Form(5),
) -> ChatResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    filename = file.filename.lower()
    is_pdf = filename.endswith(".pdf")
    is_image = filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
    content_type = (file.content_type or "").lower()

    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=400,
            detail="Only PDF or image files are supported (.pdf, .png, .jpg, .jpeg, .webp, .gif)",
        )

    if is_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            result = rag_service.chat_with_pdf_attachment(
                question=question,
                pdf_path=tmp_path,
                source_name=file.filename,
                namespace=namespace,
                top_k=top_k,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        if content_type and not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not recognized as an image")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        result = rag_service.chat_with_image_attachment(
            question=question,
            image_bytes=image_bytes,
            image_mime_type=content_type or "image/jpeg",
            source_name=file.filename,
            namespace=namespace,
            top_k=top_k,
        )

    return _build_chat_response(result)
