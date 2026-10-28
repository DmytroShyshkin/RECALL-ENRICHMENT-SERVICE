# api.py
from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIError, RateLimitError
from pydantic import BaseModel

import main

app = FastAPI()
client = main.build_client()
provider: main.EnrichmentProvider = main.GroqProvider(client)


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
        result = provider.enrich(request.word, request.source_language, request.target_language)
    except main.EnrichmentValidationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Service unavailable. Please try again later.")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return EnrichmentResponse(translation=result.translation, description=result.description)
