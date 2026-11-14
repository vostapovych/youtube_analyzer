"""VADER baseline (replicates last year's approach)."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = None


def label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def vader_sentiment(text: str) -> str:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return label_from_compound(_analyzer.polarity_scores(text or "")["compound"])


def vader_batch(texts: list[str]) -> list[str]:
    return [vader_sentiment(t) for t in texts]
