"""Offline tests — all external calls are mocked."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from commentlens import analysis, config, gemini_classifier as gc, youtube_client as yc
from commentlens.vader_baseline import label_from_compound, vader_sentiment


# ---------- extract_video_id ----------
@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=w224tgfrTBY",
    "https://www.youtube.com/watch?feature=share&v=w224tgfrTBY&t=10",
    "https://youtu.be/w224tgfrTBY?si=abc",
    "https://www.youtube.com/shorts/w224tgfrTBY",
    "w224tgfrTBY",
])
def test_extract_video_id_ok(url):
    assert yc.extract_video_id(url) == "w224tgfrTBY"


@pytest.mark.parametrize("bad", ["", "hello", "https://example.com/watch?v=123", "https://youtube.com/"])
def test_extract_video_id_invalid(bad):
    with pytest.raises(yc.YouTubeError):
        yc.extract_video_id(bad)


# ---------- YouTube fakes ----------
class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


def _page(n, start, token):
    items = [{"id": f"c{start + i}", "snippet": {"totalReplyCount": 0, "topLevelComment": {"snippet": {
        "authorDisplayName": "a", "textDisplay": f"text {start + i}", "likeCount": 0, "publishedAt": ""}}}}
        for i in range(n)]
    d = {"items": items}
    if token:
        d["nextPageToken"] = token
    return d


class FakeSession:
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return self.pages.pop(0)


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    monkeypatch.setattr(config, "YOUTUBE_API_KEY", "test")


def test_pagination_stops_at_max():
    s = FakeSession([FakeResp(200, _page(100, 0, "t1")), FakeResp(200, _page(100, 100, "t2")),
                     FakeResp(200, _page(50, 200, "t3"))])
    out = yc.fetch_comments("w224tgfrTBY", 250, session=s)
    assert len(out) == 250
    assert len(s.calls) == 3
    assert s.calls[2]["maxResults"] == 50 and s.calls[1]["pageToken"] == "t1"


def test_pagination_stops_without_next_token():
    s = FakeSession([FakeResp(200, _page(100, 0, "t1")), FakeResp(200, _page(30, 100, None))])
    out = yc.fetch_comments("w224tgfrTBY", 500, session=s)
    assert len(out) == 130 and len(s.calls) == 2


def _err(status, reason):
    return FakeResp(status, {"error": {"code": status, "errors": [{"reason": reason}]}})


@pytest.mark.parametrize("reason,msg", [
    ("commentsDisabled", "вимкнені"),
    ("quotaExceeded", "квоту"),
    ("videoNotFound", "не знайдено"),
])
def test_error_mapping(reason, msg):
    s = FakeSession([_err(403 if reason != "videoNotFound" else 404, reason)])
    with pytest.raises(yc.YouTubeError, match=msg):
        yc.fetch_comments("w224tgfrTBY", 100, session=s)


def test_video_not_found_empty_items():
    with pytest.raises(yc.YouTubeError, match="не знайдено"):
        yc.fetch_video_info("w224tgfrTBY", session=FakeSession([FakeResp(200, {"items": []})]))


# ---------- Gemini ----------
def test_make_batches():
    b = gc.make_batches(list(range(120)), 50)
    assert [len(x) for x in b] == [50, 50, 20]
    assert sum(b, []) == list(range(120))


class FakeModels:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _resp(items):
    return SimpleNamespace(parsed=None, text=json.dumps({"items": items}),
                           usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5,
                                                          total_token_count=15))


def _item(i, s="positive"):
    return {"idx": i, "sentiment": s, "topic": "тема", "is_question": False, "is_toxic": False}


def test_malformed_batch_retry_then_unknown():
    bad = SimpleNamespace(parsed=None, text="{not json", usage_metadata=None)
    partial = _resp([_item(0)])  # item 1 missing
    client = SimpleNamespace(models=FakeModels([bad, partial]))
    labels, stats = gc.classify_comments(["a", "b"], client=client, model="m", pause=0, sleep=lambda s: None)
    assert client.models.calls == 2
    assert labels[0]["sentiment"] == "positive"
    assert labels[1]["sentiment"] == "unknown"
    assert stats.requests == 2 and stats.total_tokens == 15


def test_full_batch_single_call():
    client = SimpleNamespace(models=FakeModels([_resp([_item(1, "negative"), _item(0)])]))
    labels, stats = gc.classify_comments(["a", "b"], client=client, model="m", pause=0, sleep=lambda s: None)
    assert [l["sentiment"] for l in labels] == ["positive", "negative"]
    assert stats.requests == 1


class RateLimit(Exception):
    code = 429

    def __str__(self):
        return "429 RESOURCE_EXHAUSTED {'retryDelay': '7s'}"


def test_retry_on_429_respects_delay():
    slept = []
    calls = iter([RateLimit(), "ok"])

    def fn():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    assert gc.call_with_retry(fn, sleep=slept.append) == "ok"
    assert slept and slept[0] >= 7


# ---------- VADER ----------
def test_vader_thresholds():
    assert label_from_compound(0.05) == "positive"
    assert label_from_compound(-0.05) == "negative"
    assert label_from_compound(0.0) == "neutral"
    assert label_from_compound(0.049) == "neutral"
    assert vader_sentiment("I love this, amazing video!") == "positive"
    assert vader_sentiment("This is terrible and awful") == "negative"


# ---------- aggregation & metrics ----------
def test_aggregate_percentages():
    comments = [{"id": str(i), "text": "t", "likes": i} for i in range(4)]
    labels = [{"sentiment": s, "topic": t, "is_question": q, "is_toxic": x} for s, t, q, x in [
        ("positive", "Ціна", True, False), ("positive", "ціна", False, False),
        ("negative", "якість", False, True), ("unknown", "unknown", False, False)]]
    df = analysis.build_dataframe(comments, labels, ["positive", "neutral", "negative", "neutral"])
    a = analysis.aggregate(df)
    assert a["total"] == 4 and a["unknown"] == 1
    assert a["gemini_pct"] == {"positive": 66.7, "neutral": 0.0, "negative": 33.3}
    assert a["vader_pct"] == {"positive": 25.0, "neutral": 50.0, "negative": 25.0}
    assert a["questions"] == 1 and a["toxic"] == 1
    assert a["top_topics"] == {"ціна": 2, "якість": 1}


def test_compute_metrics_small_example():
    y_true = ["positive", "positive", "neutral", "negative"]
    y_pred = ["positive", "neutral", "neutral", "negative"]
    m = analysis.compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["positive"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert m["per_class"]["neutral"]["precision"] == 0.5
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_sample_for_labeling_skips_junk_and_balances():
    per = {"v1": ["Good video", "good  video", "😀😀", "", "Nice one", "ok?"],
           "v2": ["Погане відео", "Класно", "👍", "Дякую!"]}
    rows = analysis.sample_for_labeling(per, 4, seed=1)
    assert [r["video_id"] for r in rows].count("v1") == 2
    assert [r["video_id"] for r in rows].count("v2") == 2
    texts = [r["text"] for r in rows]
    assert "😀😀" not in texts and "👍" not in texts and "" not in texts
    assert all(r["label"] == "" for r in rows) and [r["id"] for r in rows] == [1, 2, 3, 4]


def test_sample_for_labeling_redistributes_small_video():
    per = {"small": ["one comment", "two comment"],
           "a": [f"comment a{i}" for i in range(50)], "b": [f"comment b{i}" for i in range(50)]}
    rows = analysis.sample_for_labeling(per, 30, seed=0)
    vids = [r["video_id"] for r in rows]
    assert len(rows) == 30 and vids.count("small") == 2
    assert vids.count("a") == 14 and vids.count("b") == 14
