# api.py
from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIError
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
    try:
        result = main.enrich_word(client, request.word, request.source_language, request.target_language)
    except main.EnrichmentValidationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Service unavailable. Please try again later.")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return EnrichmentResponse(translation=result.translation, description=result.description)