import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class WordEnrichment(BaseModel):
    translation: str
    description: str

def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ["GROQ_API_KEY"]

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

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
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b"
        , messages=messages
        , response_format={"type": "json_object"}
    )
    
    response_content = response.choices[0].message.content
    
    return WordEnrichment.model_validate_json(response_content)

if __name__ == "__main__":
    client = build_client()
    result = enrich_word(client, word="звичайно", source_language="ua", target_language="en")
    print(result)