# api.py
from fastapi import FastAPI
from pydantic import BaseModel

import main

app = FastAPI()
client = main.build_client()


class EnrichmentRequest(BaseModel):
    word: str
    source_language: str
    target_language: str


class EnrichmentResponse(BaseModel):
    translation: str
    description: str


@app.post("/enrich", response_model=EnrichmentResponse)
def enrich(request: EnrichmentRequest) -> EnrichmentResponse:
    result = main.enrich_word(client, request.word, request.source_language, request.target_language)
    return EnrichmentResponse(translation=result.translation, description=result.description)