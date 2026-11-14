"""Tiny JSON cache in data/cache/."""
import json
import re

from . import config


def _path(kind: str, *parts):
    key = "_".join(re.sub(r"[^A-Za-z0-9._-]", "-", str(p)) for p in parts)
    return config.CACHE_DIR / f"{kind}_{key}.json"


def load(kind: str, *parts):
    p = _path(kind, *parts)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save(kind: str, data, *parts) -> None:
    p = _path(kind, *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
