import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class WordEnrichment(BaseModel):
    examples: list[str]
    part_of_speech: str
    usage_note: str

def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ["GROQ_API_KEY"]

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

SYSTEM_PROMPT = (
    "You are a vocabulary enrichment assistant for a language-learning app. "
    "Given a word and a source/target language pair, mentally translate the word "
    "and produce a short enrichment about its use in the target language. "
    "Respond with nothing but a single raw JSON object — no markdown, no code fences, no extra text. "
    "The JSON object must contain exactly these fields and nothing else:\n"
    "  \"examples\": an array of 2 to 3 short example sentences in the target language, "
    "each naturally using the word's translation\n"
    "  \"part_of_speech\": a single lowercase word, e.g. \"noun\", \"verb\", \"adjective\", \"adverb\"\n"
    "  \"usage_note\": one short sentence (under 20 words) about register, common context, or a usage tip"
)


def build_user_prompt(word: str, source_language: str, target_language: str) -> str:
    return (
        f"Word: \"{word}\"\n"
        f"Source language: {source_language}\n"
        f"Target language: {target_language}\n"
        "Produce the enrichment JSON described above for this word's translation into the target language."
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