# api.py
import os
import secrets

from fastapi import FastAPI, HTTPException, Header, Depends
from openai import APIConnectionError, APIError, RateLimitError
from pydantic import BaseModel

import main

app = FastAPI()
client = main.build_client()
provider: main.EnrichmentProvider = main.GroqProvider(client)

INTERNAL_SHARED_SECRET = os.environ.get("INTERNAL_SHARED_SECRET")
if not INTERNAL_SHARED_SECRET:
    raise RuntimeError("INTERNAL_SHARED_SECRET environment variable is not set")


def verify_internal_secret(x_internal_secret: str = Header(...)) -> None:
    """Rejects any request that doesn't carry the shared secret RECALL-API
    sends on every call. Takes the place of the AWS security group's
    IP allowlist, which Render's free plan doesn't offer."""
    if not secrets.compare_digest(x_internal_secret, INTERNAL_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Invalid or missing internal secret")


class EnrichmentRequest(BaseModel):
    word: str
    source_language: str
    target_language: str


class EnrichmentResponse(BaseModel):
    translation: str
    description: str


@app.post("/enrich", response_model=EnrichmentResponse, dependencies=[Depends(verify_internal_secret)])
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
