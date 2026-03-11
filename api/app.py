from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieval.search_code import search_code, explain_code

app = FastAPI(title="Codebase Assistant")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask-code", response_model=AskResponse)
def ask_code(request: AskRequest):

    results = search_code(request.question)

    answer = explain_code(request.question, results)

    sources = [f"{file_path} chunk {chunk_index}" for file_path, chunk_index, _ in results]

    return AskResponse(answer=answer, sources=sources)
