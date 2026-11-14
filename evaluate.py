"""python evaluate.py data/labeling/labeled.csv
Compares Gemini vs VADER against manual labels."""
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from commentlens import analysis, config, gemini_classifier  # noqa: E402
from commentlens.vader_baseline import vader_batch  # noqa: E402

LABELS = analysis.LABELS


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    vmax = max(max(r) for r in cm) or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="white" if cm[i][j] > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("labeled_csv")
    p.add_argument("--model", default=config.GEMINI_MODEL)
    args = p.parse_args()

    df = pd.read_csv(args.labeled_csv, encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    print(f"Labeled rows: {len(df)}")
    texts = df["text"].astype(str).tolist()

    labels, gstats = gemini_classifier.classify_comments(texts, "", args.model)
    df["gemini"] = [l["sentiment"] for l in labels]
    df["vader"] = vader_batch(texts)

    m_g = analysis.compute_metrics(df["label"].tolist(), df["gemini"].tolist())
    m_v = analysis.compute_metrics(df["label"].tolist(), df["vader"].tolist())
    out = {"n": len(df), "model": args.model, "gemini": m_g, "vader": m_v,
           "gemini_unknown": int((df["gemini"] == "unknown").sum()),
           "gemini_seconds": round(gstats.seconds, 2), "gemini_requests": gstats.requests,
           "gemini_tokens": gstats.total_tokens}

    R = config.RESULTS_DIR
    R.mkdir(exist_ok=True)
    (R / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    plot_confusion(m_g["confusion_matrix"], f"Gemini ({args.model})", R / "confusion_gemini.png")
    plot_confusion(m_v["confusion_matrix"], "VADER", R / "confusion_vader.png")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = range(2)
    ax.bar([i - 0.2 for i in x], [m_g["accuracy"], m_g["macro_f1"]], 0.4, label="Gemini", color="#4361ee")
    ax.bar([i + 0.2 for i in x], [m_v["accuracy"], m_v["macro_f1"]], 0.4, label="VADER", color="#f4a261")
    ax.set_xticks(list(x), ["Accuracy", "Macro-F1"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Gemini vs VADER (n={len(df)})")
    fig.tight_layout()
    fig.savefig(R / "accuracy_f1.png", dpi=150)
    plt.close(fig)

    lines = [f"| Метрика | Gemini ({args.model}) | VADER |", "|---|---|---|",
             f"| Accuracy | {m_g['accuracy']:.3f} | {m_v['accuracy']:.3f} |",
             f"| Macro-F1 | {m_g['macro_f1']:.3f} | {m_v['macro_f1']:.3f} |"]
    for lab in LABELS:
        g, v = m_g["per_class"][lab], m_v["per_class"][lab]
        lines.append(f"| F1 {lab} (P/R) | {g['f1']:.2f} ({g['precision']:.2f}/{g['recall']:.2f}) | "
                     f"{v['f1']:.2f} ({v['precision']:.2f}/{v['recall']:.2f}) |")
    lines.append(f"\nn = {len(df)}; Gemini: {gstats.requests} запитів, {gstats.seconds:.1f} с, "
                 f"{gstats.total_tokens} токенів.")
    (R / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # per-video breakdown (different languages / audiences)
    per_video = {}
    for vid, g in df.groupby("video_id"):
        per_video[vid] = {"n": len(g), "gemini_acc": round(float((g["gemini"] == g["label"]).mean()), 3),
                          "vader_acc": round(float((g["vader"] == g["label"]).mean()), 3)}
    out["per_video"] = per_video
    (R / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines.append("\n| Відео | n | Gemini acc | VADER acc |\n|---|---|---|---|")
    lines += [f"| {vid} | {p['n']} | {p['gemini_acc']:.2f} | {p['vader_acc']:.2f} |" for vid, p in per_video.items()]
    (R / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df[["id", "video_id", "text", "label", "gemini", "vader"]].to_csv(
        R / "predictions.csv", index=False, encoding="utf-8-sig")

    dis = df[(df["vader"] != df["label"]) & (df["gemini"] == df["label"])]
    dis[["id", "video_id", "text", "label", "gemini", "vader"]].to_csv(
        R / "disagreements.csv", index=False, encoding="utf-8-sig")

    print("\n".join(lines))
    print(f"\nDisagreements (VADER wrong, Gemini right): {len(dis)} -> results/disagreements.csv")


if __name__ == "__main__":
    main()
