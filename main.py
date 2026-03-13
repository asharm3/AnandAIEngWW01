from datetime import datetime, timezone
import json
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


# region agent log
def _agent_log(hypothesis_id: str, message: str, data: dict) -> None:
    try:
        log_entry = {
            "sessionId": "c656de",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": "main.py:summarize",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-c656de.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        # logging must never break the endpoint
        pass
# endregion agent log

_SYSTEM_PROMPT = """You are a summarization assistant.
Summarize the user-provided text faithfully and concisely.
Constraints:
- Do not add new facts or assumptions.
- Keep the summary within the requested maximum length.
- No preamble, no explanations, no analysis.
- Output MUST be plain text only (no markdown, no quotes).
- If the input is empty, output an empty string."""


class SummarizeRequest(BaseModel):
    text: str
    max_length: int


@app.post("/summarize")
def summarize(payload: SummarizeRequest):
    text = payload.text or ""
    max_length = payload.max_length

    if max_length <= 0:
        raise HTTPException(status_code=400, detail="max_length must be > 0")

    if not text.strip():
        return {"summary": ""}

    api_key = os.getenv("OPENAI_API_KEY")
    _agent_log("H1", "loaded_api_key", {"has_key": bool(api_key)})
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured",
        )

    client = OpenAI(api_key=api_key)
    _agent_log("H2", "before_openai_call", {"max_length": max_length, "text_len": len(text)})
    user_prompt = f"Summarize the following text in at most {max_length} characters.\n\nText:\n{text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        _agent_log("H3", "openai_call_success", {})
    except Exception as exc:
        _agent_log("H4", "openai_call_error", {"error_type": type(exc).__name__, "error_str": str(exc)})
        raise

    summary = (resp.choices[0].message.content or "").strip()
    if len(summary) > max_length:
        summary = summary[:max_length].rstrip()

    return {"summary": summary}


_SENTIMENT_PROMPT = """You are a sentiment analyst. For the user's text, respond with only a single JSON object (no markdown, no other text) with exactly these keys: "sentiment" (one of: positive, negative, neutral), "confidence" (number between 0 and 1), "explanation" (short string)."""


class SentimentRequest(BaseModel):
    text: str


@app.post("/analyze-sentiment")
def analyze_sentiment(payload: SentimentRequest):
    text = (payload.text or "").strip()
    if not text:
        return {"sentiment": "neutral", "confidence": 0.0, "explanation": "No text provided."}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _SENTIMENT_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        out = json.loads(raw)
        sentiment = out.get("sentiment", "neutral")
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"
        return {
            "sentiment": sentiment,
            "confidence": float(out.get("confidence", 0.0)),
            "explanation": str(out.get("explanation", "")),
        }
    except (json.JSONDecodeError, TypeError):
        return {"sentiment": "neutral", "confidence": 0.0, "explanation": raw or "Could not parse model response."}


@app.get("/health")
def health():
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"status": "ok", "timestamp": timestamp}

