"""Manual labeling tool: streamlit run label_tool.py --server.port 8502

Shows data/labeling/to_label.csv one comment at a time. Every click is saved to
data/labeling/labeled.csv (UTF-8); on restart it resumes from the first unlabeled row.
Paths can be overridden with LABEL_IN / LABEL_OUT env vars (used for testing).
"""
import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
IN_PATH = Path(os.getenv("LABEL_IN", ROOT / "data" / "labeling" / "to_label.csv"))
OUT_PATH = Path(os.getenv("LABEL_OUT", ROOT / "data" / "labeling" / "labeled.csv"))
LABELS = [("positive", "😊 positive"), ("neutral", "😐 neutral"), ("negative", "😠 negative")]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def load() -> pd.DataFrame:
    """to_label.csv rows, with labels already saved in labeled.csv merged in by id."""
    df = _read(IN_PATH)
    df["label"] = ""
    if OUT_PATH.exists():
        saved = _read(OUT_PATH).set_index("id")["label"]
        df["label"] = df["id"].map(saved).fillna("")
    return df


def save(df: pd.DataFrame) -> None:
    tmp = OUT_PATH.with_suffix(".tmp")
    df[["id", "video_id", "text", "label"]].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, OUT_PATH)  # atomic: a crash never leaves a half-written file


def first_unlabeled(df: pd.DataFrame) -> int:
    empty = df.index[df["label"] == ""]
    return int(empty[0]) if len(empty) else len(df)


st.set_page_config(page_title="Label tool", page_icon="🏷️", layout="centered")
st.markdown("<style>div.stButton > button {height: 4.5rem; font-size: 1.25rem;}</style>",
            unsafe_allow_html=True)

if "df" not in st.session_state:
    st.session_state.df = load()
    st.session_state.idx = first_unlabeled(st.session_state.df)
df: pd.DataFrame = st.session_state.df
n = len(df)
done = int((df["label"] != "").sum())


def set_label(label: str) -> None:
    i = st.session_state.idx
    df.at[i, "label"] = label
    try:
        save(df)
        st.session_state.save_error = None
    except PermissionError:
        st.session_state.save_error = f"Не вдалося зберегти {OUT_PATH.name} (файл відкрито в Excel?). Закрийте його й натисніть ще раз."
        return
    # go to the next unlabeled row after this one (or the end)
    nxt = df.index[(df.index > i) & (df["label"] == "")]
    st.session_state.idx = int(nxt[0]) if len(nxt) else first_unlabeled(df)


def back() -> None:
    st.session_state.idx = max(0, st.session_state.idx - 1)


st.title("🏷️ Розмітка коментарів")
st.progress(done / n if n else 1.0, text=f"Розмічено {done} / {n}")
if st.session_state.get("save_error"):
    st.error(st.session_state.save_error)

idx = st.session_state.idx
if idx >= n:
    st.success(f"Готово! Усі {n} коментарів розмічено й збережено в `{OUT_PATH.relative_to(ROOT) if OUT_PATH.is_relative_to(ROOT) else OUT_PATH}`.")
    st.button("⬅️ Назад", on_click=back, disabled=n == 0)
else:
    row = df.iloc[idx]
    st.caption(f"Рядок {idx + 1} з {n} · id {row['id']} · відео {row['video_id']}")
    with st.container(border=True):
        # escaped so comment content can't inject markup
        st.markdown(f"<div style='font-size:1.3rem; line-height:1.5; white-space:pre-wrap'>"
                    f"{html.escape(row['text'])}</div>", unsafe_allow_html=True)
    if row["label"]:
        st.info(f"Поточна мітка: **{row['label']}** (натисніть іншу кнопку, щоб змінити)")
    cols = st.columns(3)
    for col, (lab, title) in zip(cols, LABELS):
        col.button(title, key=f"b_{lab}", on_click=set_label, args=(lab,), width="stretch",
                   type="primary" if row["label"] == lab else "secondary")
    st.button("⬅️ Назад", on_click=back, disabled=idx == 0)
