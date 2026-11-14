"""Loads settings from .env and defines project-wide constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL = "gemini-3.6-flash"  # chosen via client.models.list() on 2026-11-14
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

DEFAULT_MAX_COMMENTS = 300
MAX_COMMENTS_LIMIT = 500
BATCH_SIZE = 50
BATCH_PAUSE_SEC = 1.0  # pause between Gemini batches (free-tier RPM)

CACHE_DIR = ROOT / "data" / "cache"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = ROOT / "results"
LABELING_DIR = ROOT / "data" / "labeling"

SENTIMENTS = ("positive", "neutral", "negative")
