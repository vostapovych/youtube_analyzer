"""Aggregation, dataframe building, metrics, and the end-to-end pipeline."""
import random
import time

import pandas as pd

from . import cache, config, gemini_classifier, youtube_client
from .vader_baseline import vader_batch

LABELS = list(config.SENTIMENTS)


def build_dataframe(comments: list[dict], gemini_labels: list[dict], vader_labels: list[str]) -> pd.DataFrame:
    rows = []
    for c, g, v in zip(comments, gemini_labels, vader_labels):
        rows.append({**c, **g, "vader_sentiment": v})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["id", "author", "text", "likes", "sentiment", "topic",
                                   "is_question", "is_toxic", "vader_sentiment"])
    return df


def percentages(series: pd.Series, labels=LABELS) -> dict:
    n = len(series)
    counts = series.value_counts()
    return {lab: (round(100.0 * counts.get(lab, 0) / n, 1) if n else 0.0) for lab in labels}


def aggregate(df: pd.DataFrame, top_n_topics: int = 10) -> dict:
    known = df[df["sentiment"].isin(LABELS)] if len(df) else df
    return {
        "total": int(len(df)),
        "classified": int(len(known)),
        "unknown": int(len(df) - len(known)),
        "gemini_pct": percentages(known["sentiment"]) if len(known) else {l: 0.0 for l in LABELS},
        "vader_pct": percentages(df["vader_sentiment"]) if len(df) else {l: 0.0 for l in LABELS},
        "questions": int(df["is_question"].sum()) if len(df) else 0,
        "toxic": int(df["is_toxic"].sum()) if len(df) else 0,
        "top_topics": (known[known["topic"] != "unknown"]["topic"].str.strip().str.lower()
                       .value_counts().head(top_n_topics).to_dict()) if len(known) else {},
    }


def compute_metrics(y_true: list[str], y_pred: list[str], labels=LABELS) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def summary_sample(df: pd.DataFrame, k: int = 80, seed: int = 0) -> list[str]:
    """Top-liked comments + random rest, so the summary sees both popular and typical views."""
    if df.empty:
        return []
    top = df.sort_values("likes", ascending=False).head(k // 2)
    rest = df.drop(top.index)
    rnd = rest.sample(min(len(rest), k - len(top)), random_state=seed) if len(rest) else rest
    return list(pd.concat([top, rnd])["text"])


def run_analysis(url: str, max_comments: int = config.DEFAULT_MAX_COMMENTS, model: str | None = None,
                 use_cache: bool = True, progress=None) -> dict:
    """Full pipeline: YouTube -> Gemini + VADER -> aggregation -> Gemini summary."""
    model = model or config.GEMINI_MODEL
    t_start = time.perf_counter()
    video_id = youtube_client.extract_video_id(url)

    # 1. YouTube
    t0 = time.perf_counter()
    yt = cache.load("youtube", video_id, max_comments) if use_cache else None
    yt_cached = yt is not None
    if yt is None:
        info = youtube_client.fetch_video_info(video_id)
        comments = youtube_client.fetch_comments(video_id, max_comments)
        yt = {"info": info, "comments": comments}
        if comments:
            cache.save("youtube", yt, video_id, max_comments)
    yt_seconds = time.perf_counter() - t0
    if not yt["comments"]:
        raise youtube_client.YouTubeError(youtube_client.ERROR_MESSAGES["no_comments"])
    info, comments = yt["info"], yt["comments"]
    texts = [c["text"] for c in comments]

    # 2. Gemini classification + summary (cached together)
    gm = cache.load("gemini", video_id, model, max_comments) if use_cache else None
    gm_cached = gm is not None
    vader = vader_batch(texts)
    if gm is None:
        client = gemini_classifier.make_client()
        labels, gstats = gemini_classifier.classify_comments(texts, info["title"], model, client,
                                                             progress=progress)
        df = build_dataframe(comments, labels, vader)
        agg = aggregate(df)
        summary = gemini_classifier.summarize(agg, summary_sample(df), info["title"], model, client, gstats)
        gm = {"labels": labels, "summary": summary, "stats": gstats.as_dict()}
        if agg["unknown"] < len(df):  # don't cache total failures
            cache.save("gemini", gm, video_id, model, max_comments)
    df = build_dataframe(comments, gm["labels"], vader)
    agg = aggregate(df)

    return {
        "video": info,
        "model": model,
        "df": df,
        "agg": agg,
        "summary": gm["summary"],
        "timings": {
            "youtube_sec": round(yt_seconds, 2),
            "gemini_sec": round(gm["stats"]["seconds"], 2),
            "total_sec": round(time.perf_counter() - t_start, 2),
            "youtube_cached": yt_cached,
            "gemini_cached": gm_cached,
        },
        "gemini_stats": gm["stats"],
    }


def report_json(result: dict) -> dict:
    return {
        "video": result["video"], "model": result["model"], "aggregate": result["agg"],
        "summary": result["summary"], "timings": result["timings"], "gemini_stats": result["gemini_stats"],
        "comments": result["df"].to_dict(orient="records"),
    }


def sample_for_labeling(per_video: dict[str, list[str]], n: int, seed: int = 42) -> list[dict]:
    """Pick ~n/len(videos) unique, non-trivial comments per video."""
    import re
    rng = random.Random(seed)
    seen, pools = set(), {}
    for vid, texts in per_video.items():
        pool = []
        for t in texts:
            norm = re.sub(r"\s+", " ", t).strip().lower()
            if not re.search(r"\w", norm) or norm in seen or len(norm) < 3:
                continue  # empty / emoji-only / duplicate
            seen.add(norm)
            pool.append(t.strip())
        rng.shuffle(pool)
        pools[vid] = pool
    # equal quotas; if a video has too few comments, its leftover goes to the others
    quota = {vid: 0 for vid in pools}
    remaining = n
    while remaining > 0:
        open_vids = [v for v in pools if quota[v] < len(pools[v])]
        if not open_vids:
            break
        share = max(1, remaining // len(open_vids))
        for v in open_vids:
            add = min(share, len(pools[v]) - quota[v], remaining)
            quota[v] += add
            remaining -= add
            if remaining == 0:
                break
    rows = [{"video_id": vid, "text": t} for vid in pools for t in pools[vid][:quota[vid]]]
    for i, r in enumerate(rows, 1):
        r["id"] = i
        r["label"] = ""
    return [{"id": r["id"], "video_id": r["video_id"], "text": r["text"], "label": ""} for r in rows]
