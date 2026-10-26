from asyncio.log import logger
from dataclasses import Field
import os
import logging

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


class WordEnrichment(BaseModel):
    translation: str = Field(min_length=1)
    description: str = Field(min_length=1)

def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ["GROQ_API_KEY"]

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    
    return OpenAI(
        api_key=api_key
        , base_url="https://api.groq.com/openai/v1"
        , timeout=30
        )

MAX_VALIDATION_ATTEMPTS=3
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a vocabulary translation assistant for a language-learning app. "
    "Given a word in its source language and a target language, translate "
    "the word into the target language and write a short description that "
    "helps a learner use the translated word correctly. "
    "Write the description in the TARGET language. "
    "Respond with nothing but a single raw JSON object — no markdown, no code "
    "fences, no extra text. The JSON object must contain exactly these fields "
    "and nothing else:\n"
    "  \"translation\": the word translated into the target language — a "
    "single word or short phrase, not a full sentence\n"
    "  \"description\": one to two short sentences, written in the target "
    "language, explaining the meaning, nuance, or typical usage context of "
    "the translated word (formality, common contexts, false-friend warnings, "
    "etc.)"
)

class EnrichmentValidationError(Exception):
    """Raised when Groq returns a 200 response whose content doesn't pass
    Pydantic validation (invalid/malformed JSON) even after retries."""
    pass

def build_user_prompt(word: str, source_language: str, target_language: str) -> str:
    return (
        f"Word: \"{word}\"\n"
        f"Source language: {source_language}\n"
        f"Target language: {target_language}\n"
        "Produce the translation and description JSON described above."
    )

def enrich_word(client: OpenAI, word: str, source_language: str, target_language: str) -> WordEnrichment:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(word, source_language, target_language)}
    ]
    
    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content

        try:
            return WordEnrichment.model_validate_json(response_content)
        except ValidationError as e:
            logger.warning(
                "Enrichment validation failed on attempt %d/%d for word=%r: %s",
                attempt, MAX_VALIDATION_ATTEMPTS, word, e
            )

    raise EnrichmentValidationError(
        f"Groq returned an invalid response for word={word!r} after "
        f"{MAX_VALIDATION_ATTEMPTS} attempts"
    )

if __name__ == "__main__":
    client = build_client()
    result = enrich_word(client, word="звичайно", source_language="ua", target_language="en")
    print(result)