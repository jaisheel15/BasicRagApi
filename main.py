import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag import doc_loader, generate_answer, text_splitting, vector_store, load_vector_store


app = FastAPI(title="RAG Backend")
_vector_store = None


class IngestRequest(BaseModel):
    pass


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_documents(file: UploadFile = File(...)):
    global _vector_store

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        docs = doc_loader(temp_path)
    finally:
        os.unlink(temp_path)

    if not docs:
        raise HTTPException(status_code=404, detail="No content could be extracted from the uploaded PDF")

    chunks = text_splitting(docs)
    _vector_store = vector_store(chunks)

    return {"documents": len(docs), "chunks": len(chunks)}


@app.post("/query")
def query_documents(request: QueryRequest):
    global _vector_store

    if _vector_store is None:
        _vector_store = load_vector_store()

    return generate_answer(_vector_store, request.question)


