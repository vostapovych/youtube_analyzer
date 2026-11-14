"""Streamlit UI: streamlit run app.py"""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from commentlens import analysis, config
from commentlens.youtube_client import YouTubeError

st.set_page_config(page_title="CommentLens", page_icon="💬", layout="wide")
st.title("💬 CommentLens — аналіз коментарів YouTube")
st.caption(f"YouTube Data API v3 + Gemini API ({config.GEMINI_MODEL}) · базова лінія: VADER")

SENT_UA = {"positive": "позитивні", "neutral": "нейтральні", "negative": "негативні", "unknown": "невідомо"}
COLORS = {"positive": "#2e9e5b", "neutral": "#8a8f98", "negative": "#d64545"}

with st.form("input"):
    url = st.text_input("Посилання на YouTube-відео", placeholder="https://www.youtube.com/watch?v=...")
    n = st.slider("Кількість коментарів", 100, config.MAX_COMMENTS_LIMIT, config.DEFAULT_MAX_COMMENTS, step=50)
    go = st.form_submit_button("Аналізувати", type="primary")

if go:
    bar = st.progress(0.0, text="Завантаження коментарів…")
    try:
        res = analysis.run_analysis(url, n, progress=lambda i, k: bar.progress(i / k, text=f"Gemini: батч {i}/{k}"))
        st.session_state["res"] = res
    except YouTubeError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.error(f"Неочікувана помилка: {type(e).__name__}: {e}")
    bar.empty()

res = st.session_state.get("res")
if res:
    v, a, t, df = res["video"], res["agg"], res["timings"], res["df"]
    st.subheader(v["title"])
    st.write(f"**{v['channel']}** · 👁 {v['views']:,} переглядів · 👍 {v['likes']:,} · 💬 {v['comment_count']:,} коментарів"
             f" · [відкрити](https://www.youtube.com/watch?v={v['video_id']})")

    c = st.columns(6)
    c[0].metric("Проаналізовано", a["total"])
    c[1].metric("Позитивні", f"{a['gemini_pct']['positive']}%")
    c[2].metric("Нейтральні", f"{a['gemini_pct']['neutral']}%")
    c[3].metric("Негативні", f"{a['gemini_pct']['negative']}%")
    c[4].metric("Питання", a["questions"])
    c[5].metric("Токсичні", a["toxic"])
    g = res["gemini_stats"]
    st.caption(f"⏱ YouTube: {t['youtube_sec']} с{' (кеш)' if t['youtube_cached'] else ''} · "
               f"Gemini: {t['gemini_sec']} с{' (кеш)' if t['gemini_cached'] else ''} · всього: {t['total_sec']} с · "
               f"запитів до Gemini: {g['requests']} · токенів: {g['total_tokens']:,}"
               + (f" · не класифіковано: {a['unknown']}" if a["unknown"] else ""))

    left, right = st.columns(2)
    with left:
        comp = pd.DataFrame([{"метод": m, "тональність": SENT_UA[s], "відсоток": a[k][s]}
                             for m, k in (("Gemini", "gemini_pct"), ("VADER", "vader_pct"))
                             for s in config.SENTIMENTS])
        fig = px.bar(comp, x="тональність", y="відсоток", color="метод", barmode="group",
                     title="Розподіл тональності: Gemini vs VADER",
                     color_discrete_map={"Gemini": "#4361ee", "VADER": "#f4a261"})
        st.plotly_chart(fig, width="stretch")
    with right:
        tt = pd.DataFrame(list(a["top_topics"].items()), columns=["тема", "кількість"])
        fig = px.bar(tt.iloc[::-1], x="кількість", y="тема", orientation="h", title="Топ теми")
        st.plotly_chart(fig, width="stretch")

    s1, s2 = st.columns([3, 2])
    with s1:
        st.markdown("#### 🧠 Що думає аудиторія (Gemini)")
        st.markdown("\n".join(f"- {b}" for b in res["summary"]["bullets"]))
    with s2:
        st.markdown("#### ❓ Топ питань і прохань глядачів")
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(res["summary"]["top_questions"], 1)) or "—")

    tox = df[df["is_toxic"] == True]  # noqa: E712
    with st.expander(f"☣️ Токсичні коментарі ({len(tox)})"):
        st.dataframe(tox[["author", "text", "likes"]], width="stretch", hide_index=True)

    st.markdown("#### Усі коментарі")
    f1, f2 = st.columns(2)
    sent_f = f1.multiselect("Тональність", sorted(df["sentiment"].unique()), format_func=lambda s: SENT_UA.get(s, s))
    topic_f = f2.multiselect("Тема", sorted(df["topic"].str.strip().str.lower().unique()))
    view = df
    if sent_f:
        view = view[view["sentiment"].isin(sent_f)]
    if topic_f:
        view = view[view["topic"].str.strip().str.lower().isin(topic_f)]
    st.dataframe(view[["text", "sentiment", "vader_sentiment", "topic", "is_question", "is_toxic", "likes", "author"]],
                 width="stretch", hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), f"{v['video_id']}.csv", "text/csv")
    d2.download_button("⬇️ JSON", json.dumps(analysis.report_json(res), ensure_ascii=False, indent=1),
                       f"{v['video_id']}.json", "application/json")
