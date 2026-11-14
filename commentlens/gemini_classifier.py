"""Gemini-based comment classification (structured output) + summary."""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

from . import config


# ---------- schemas ----------
class CommentLabel(BaseModel):
    idx: int
    sentiment: Literal["positive", "neutral", "negative"]
    topic: str
    is_question: bool
    is_toxic: bool


class BatchResult(BaseModel):
    items: list[CommentLabel]


class Summary(BaseModel):
    bullets: list[str]
    top_questions: list[str]


UNKNOWN = {"sentiment": "unknown", "topic": "unknown", "is_question": False, "is_toxic": False}

CLASSIFY_PROMPT = """You classify YouTube comments for the video "{title}".
For EVERY comment below return one item with the same idx:
- sentiment: positive | neutral | negative (the commenter's attitude; jokes/sarcasm judged by intent)
- topic: short topic label of 1-3 words, ALWAYS in Ukrainian (e.g. "якість відео", "ціна", "одяг")
- is_question: true if the comment asks the author a question or requests something
- is_toxic: true if insulting, hateful, harassing or obscene
Comments may be in any language; classify the original text, do not translate.
Return exactly {n} items.

Comments (JSON):
{comments}"""

SUMMARY_PROMPT = """Ти аналітик YouTube-коментарів. Відео: "{title}".
Агрегована статистика (класифікація Gemini): {stats}
Нижче вибірка коментарів (JSON). Напиши українською:
- bullets: 4-6 коротких пунктів "що думає аудиторія" (конкретно, з посиланням на теми);
- top_questions: до 5 найважливіших питань/прохань глядачів до автора (перефразуй стисло українською).

Коментарі:
{comments}"""


@dataclass
class GeminiStats:
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0
    errors: list = field(default_factory=list)

    def add_usage(self, resp) -> None:
        u = getattr(resp, "usage_metadata", None)
        if u is None:
            return
        self.prompt_tokens += u.prompt_token_count or 0
        self.output_tokens += u.candidates_token_count or 0
        self.total_tokens += u.total_token_count or 0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def make_client():
    from google import genai
    from google.genai import types
    if not config.GEMINI_API_KEY:
        raise RuntimeError("Не задано GEMINI_API_KEY у файлі .env.")
    # 120 s per request so a stuck call fails (and is retried) instead of hanging forever
    return genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(timeout=120_000))


def _gen_config(schema, model: str):
    from google.genai import types
    kwargs = dict(temperature=0, response_mime_type="application/json", response_schema=schema)
    # keep latency/cost low: minimal reasoning for classification
    if model.startswith("gemini-2.5"):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0 if "flash" in model else 128)
    elif model.startswith("gemini-3"):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
    return types.GenerateContentConfig(**kwargs)


def _retry_delay(err) -> float | None:
    """Extract the server-suggested retry delay (seconds) from an API error."""
    text = str(err)
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text) or \
        re.search(r"retry in (\d+(?:\.\d+)?)\s*s", text, re.I)
    return float(m.group(1)) if m else None


def call_with_retry(fn, max_attempts: int = 6, base_delay: float = 2.0, sleep=time.sleep):
    """Call fn(); retry on 429/5xx with exponential backoff (respecting retryDelay)."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            retriable = code == 429 or (isinstance(code, int) and 500 <= code < 600)
            if not retriable or attempt == max_attempts - 1:
                raise
            delay = _retry_delay(e) or base_delay * (2 ** attempt)
            sleep(min(delay + 0.5, 90))


def make_batches(items: list, size: int = config.BATCH_SIZE) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _parse_batch(resp, n: int) -> dict[int, dict]:
    """Parse response into {idx: label}; only keep idx in [0, n)."""
    parsed = getattr(resp, "parsed", None)
    if not isinstance(parsed, BatchResult):
        parsed = BatchResult.model_validate(json.loads(resp.text))
    out = {}
    for it in parsed.items:
        if 0 <= it.idx < n and it.idx not in out:
            out[it.idx] = it.model_dump(exclude={"idx"})
    return out


def classify_batch(client, model: str, texts: list[str], title: str, stats: GeminiStats,
                   sleep=time.sleep) -> list[dict]:
    """Classify one batch. Retries once if output is malformed/incomplete, then fills 'unknown'."""
    payload = json.dumps([{"idx": i, "text": t[:1500]} for i, t in enumerate(texts)], ensure_ascii=False)
    prompt = CLASSIFY_PROMPT.format(title=title, n=len(texts), comments=payload)
    cfg = _gen_config(BatchResult, model)
    result: dict[int, dict] = {}
    for _ in range(2):  # original attempt + one retry
        def _do():
            return client.models.generate_content(model=model, contents=prompt, config=cfg)
        try:
            resp = call_with_retry(_do, sleep=sleep)
            stats.requests += 1
            stats.add_usage(resp)
            got = _parse_batch(resp, len(texts))
            for k, v in got.items():
                result.setdefault(k, v)
        except (ValidationError, ValueError, TypeError, AttributeError) as e:
            stats.errors.append(f"malformed batch: {type(e).__name__}")
        if len(result) == len(texts):
            break
    return [result.get(i, dict(UNKNOWN)) for i in range(len(texts))]


def classify_comments(texts: list[str], title: str = "", model: str | None = None, client=None,
                      batch_size: int = config.BATCH_SIZE, pause: float = config.BATCH_PAUSE_SEC,
                      progress=None, sleep=time.sleep) -> tuple[list[dict], GeminiStats]:
    model = model or config.GEMINI_MODEL
    client = client or make_client()
    stats = GeminiStats()
    t0 = time.perf_counter()
    labels: list[dict] = []
    batches = make_batches(texts, batch_size)
    for bi, batch in enumerate(batches):
        if bi > 0 and pause:
            sleep(pause)
        labels.extend(classify_batch(client, model, batch, title, stats, sleep=sleep))
        if progress:
            progress(bi + 1, len(batches))
    stats.seconds = time.perf_counter() - t0
    return labels, stats


def summarize(agg: dict, sample: list[str], title: str = "", model: str | None = None, client=None,
              stats: GeminiStats | None = None) -> dict:
    model = model or config.GEMINI_MODEL
    client = client or make_client()
    stats = stats or GeminiStats()
    prompt = SUMMARY_PROMPT.format(
        title=title,
        stats=json.dumps(agg, ensure_ascii=False),
        comments=json.dumps([t[:500] for t in sample], ensure_ascii=False),
    )
    t0 = time.perf_counter()
    resp = call_with_retry(lambda: client.models.generate_content(
        model=model, contents=prompt, config=_gen_config(Summary, model)))
    stats.seconds += time.perf_counter() - t0
    stats.requests += 1
    stats.add_usage(resp)
    parsed = getattr(resp, "parsed", None)
    if not isinstance(parsed, Summary):
        parsed = Summary.model_validate(json.loads(resp.text))
    return {"bullets": parsed.bullets[:6], "top_questions": parsed.top_questions[:5]}
