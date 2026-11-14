"""Minimal YouTube Data API v3 client based on requests."""
import re

import requests

from . import config

API_URL = "https://www.googleapis.com/youtube/v3"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/(?:shorts|embed|live)/)([A-Za-z0-9_-]{11})"),
]


class YouTubeError(Exception):
    """Error with a user-facing message in Ukrainian."""


ERROR_MESSAGES = {
    "invalid_url": "Некоректне посилання на YouTube-відео.",
    "commentsDisabled": "Коментарі до цього відео вимкнені.",
    "quotaExceeded": "Вичерпано денну квоту YouTube Data API. Спробуйте завтра.",
    "videoNotFound": "Відео не знайдено (перевірте посилання).",
    "forbidden": "Доступ до даних відео заборонено.",
    "no_comments": "У цього відео немає коментарів.",
    "no_key": "Не задано YOUTUBE_API_KEY у файлі .env.",
}


def extract_video_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if _ID_RE.match(s):
        return s
    for pat in _URL_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    raise YouTubeError(ERROR_MESSAGES["invalid_url"])


def _get(endpoint: str, params: dict, session=None) -> dict:
    if not config.YOUTUBE_API_KEY:
        raise YouTubeError(ERROR_MESSAGES["no_key"])
    http = session or requests
    resp = http.get(f"{API_URL}/{endpoint}", params={**params, "key": config.YOUTUBE_API_KEY}, timeout=30)
    if resp.status_code != 200:
        raise map_error(resp)
    return resp.json()


def map_error(resp) -> YouTubeError:
    """Map an HTTP error response to a YouTubeError with a Ukrainian message."""
    reason = ""
    try:
        errors = resp.json().get("error", {}).get("errors", [])
        reason = errors[0].get("reason", "") if errors else ""
    except ValueError:
        pass
    if reason in ERROR_MESSAGES:
        return YouTubeError(ERROR_MESSAGES[reason])
    if resp.status_code == 404:
        return YouTubeError(ERROR_MESSAGES["videoNotFound"])
    if resp.status_code == 403:
        return YouTubeError(ERROR_MESSAGES["forbidden"])
    return YouTubeError(f"Помилка YouTube API (HTTP {resp.status_code}, {reason or 'unknown'}).")


def fetch_video_info(video_id: str, session=None) -> dict:
    data = _get("videos", {"part": "snippet,statistics", "id": video_id}, session)
    items = data.get("items", [])
    if not items:
        raise YouTubeError(ERROR_MESSAGES["videoNotFound"])
    sn, st = items[0]["snippet"], items[0].get("statistics", {})
    return {
        "video_id": video_id,
        "title": sn.get("title", ""),
        "channel": sn.get("channelTitle", ""),
        "published_at": sn.get("publishedAt", ""),
        "views": int(st.get("viewCount", 0)),
        "likes": int(st.get("likeCount", 0)),
        "comment_count": int(st.get("commentCount", 0)),
    }


def fetch_comments(video_id: str, max_comments: int = config.DEFAULT_MAX_COMMENTS, session=None) -> list[dict]:
    """Fetch up to max_comments top-level comments, following nextPageToken."""
    max_comments = max(1, min(max_comments, config.MAX_COMMENTS_LIMIT))
    comments: list[dict] = []
    page_token = None
    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
            "order": "relevance",
        }
        if page_token:
            params["pageToken"] = page_token
        data = _get("commentThreads", params, session)
        for item in data.get("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "id": item["id"],
                "author": sn.get("authorDisplayName", ""),
                "text": sn.get("textDisplay", ""),
                "likes": int(sn.get("likeCount", 0)),
                "published_at": sn.get("publishedAt", ""),
                "reply_count": int(item["snippet"].get("totalReplyCount", 0)),
            })
            if len(comments) >= max_comments:
                break
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return comments
