"""CLI: python cli.py analyze <url> --max 300"""
import argparse
import json
import sys

from commentlens import analysis, config
from commentlens.youtube_client import YouTubeError


def cmd_analyze(args) -> int:
    try:
        res = analysis.run_analysis(args.url, args.max, args.model, use_cache=not args.no_cache,
                                    progress=lambda i, n: print(f"  Gemini batch {i}/{n}", file=sys.stderr))
    except YouTubeError as e:
        print(f"Помилка: {e}", file=sys.stderr)
        return 1
    v, a, t, g = res["video"], res["agg"], res["timings"], res["gemini_stats"]
    print(f"\n{v['title']} — {v['channel']}")
    print(f"Переглядів: {v['views']:,}  Коментарів (всього): {v['comment_count']:,}  Проаналізовано: {a['total']}")
    print(f"Модель: {res['model']}")
    print("Gemini:", ", ".join(f"{k} {p}%" for k, p in a["gemini_pct"].items()), f"(unknown: {a['unknown']})")
    print("VADER: ", ", ".join(f"{k} {p}%" for k, p in a["vader_pct"].items()))
    print(f"Питань: {a['questions']}  Токсичних: {a['toxic']}")
    print("Топ теми:", ", ".join(f"{k} ({n})" for k, n in list(a["top_topics"].items())[:8]))
    print("\nЩо думає аудиторія:")
    for b in res["summary"]["bullets"]:
        print(f"  • {b}")
    print("\nТоп питання глядачів:")
    for q in res["summary"]["top_questions"]:
        print(f"  ? {q}")
    print(f"\nЧас: YouTube {t['youtube_sec']}s{' (кеш)' if t['youtube_cached'] else ''}, "
          f"Gemini {t['gemini_sec']}s{' (кеш)' if t['gemini_cached'] else ''}, всього {t['total_sec']}s")
    print(f"Gemini запитів: {g['requests']}, токенів: {g['total_tokens']} "
          f"(prompt {g['prompt_tokens']}, output {g['output_tokens']})")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    vid = v["video_id"]
    res["df"].to_csv(config.REPORTS_DIR / f"{vid}.csv", index=False, encoding="utf-8-sig")
    (config.REPORTS_DIR / f"{vid}.json").write_text(
        json.dumps(analysis.report_json(res), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nЗвіти: reports/{vid}.csv, reports/{vid}.json")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="CommentLens — аналіз коментарів YouTube")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("url")
    a.add_argument("--max", type=int, default=config.DEFAULT_MAX_COMMENTS)
    a.add_argument("--model", default=None)
    a.add_argument("--no-cache", action="store_true")
    a.set_defaults(func=cmd_analyze)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
