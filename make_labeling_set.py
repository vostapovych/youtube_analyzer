"""python make_labeling_set.py <url1> <url2> <url3> --n 100 --seed 42
Writes data/labeling/to_label.csv (id, video_id, text, label) for manual labeling."""
import argparse

import pandas as pd

from commentlens import analysis, cache, config, youtube_client


def main():
    p = argparse.ArgumentParser()
    p.add_argument("urls", nargs="+")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool", type=int, default=300, help="comments fetched per video to sample from")
    args = p.parse_args()

    per_video = {}
    for url in args.urls:
        vid = youtube_client.extract_video_id(url)
        yt = cache.load("youtube", vid, args.pool)
        if yt is None:
            yt = {"info": youtube_client.fetch_video_info(vid),
                  "comments": youtube_client.fetch_comments(vid, args.pool)}
            cache.save("youtube", yt, vid, args.pool)
        per_video[vid] = [c["text"] for c in yt["comments"]]
        print(f"{vid}: {len(per_video[vid])} comments — {yt['info']['title']}")

    rows = analysis.sample_for_labeling(per_video, args.n, args.seed)
    config.LABELING_DIR.mkdir(parents=True, exist_ok=True)
    out = config.LABELING_DIR / "to_label.csv"
    pd.DataFrame(rows, columns=["id", "video_id", "text", "label"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
