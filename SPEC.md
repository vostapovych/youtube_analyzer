# CommentLens — YouTube comment analyzer (Cloud Computing, Lab 2)

You are building a small proof-of-concept for a university Cloud Computing course (KNU, Lab 2).
Goal of the lab: integrate cloud APIs in a working PoC, then present it (what it does, which APIs,
how services interact, demo video, GitHub link). Speed matters: keep the code small, readable and
working. No over-engineering.

## Environment (already done by the user)
- Windows, VS Code, project folder with a Python venv in `.venv` (activate: `.venv\Scripts\activate`).
- `.env` in the project root with `YOUTUBE_API_KEY=...` and `GEMINI_API_KEY=...`
  (Google Cloud project `yt-comment-analyzer-514710`; YouTube key is restricted to YouTube Data API v3).
- **Security rules:** never print, log or commit keys. Make sure `.gitignore` contains `.env`, `d.txt`,
  `.venv/`, `__pycache__/`, `data/cache/`, `.pytest_cache/`. If a file `d.txt` with keys exists,
  tell the user to delete it (do not read it). Provide `.env.example` with empty values.
  Before every commit run `git status` and verify no secrets are staged.

## What the app does
Input: a YouTube video URL. Output: an analysis of its comments.
1. **YouTube Data API v3** (`commentThreads.list`, 1 quota unit per call, 100 comments per page):
   fetch up to N top-level comments (default 300, max 500) with pagination, `textFormat=plainText`,
   `order=relevance`. Also fetch video title/channel/stats via `videos.list` (1 unit).
   Use plain `requests` (no heavy client library needed).
2. **Gemini API** (`google-genai` SDK) classifies every comment, in batches of ~50 comments per request,
   using structured output (JSON schema / pydantic `response_schema`), temperature 0. Per comment:
   - `sentiment`: `positive` | `neutral` | `negative`
   - `topic`: short topic label (1–3 words, in Ukrainian)
   - `is_question`: bool (question addressed to the author / asking for something)
   - `is_toxic`: bool
   Comments can be in any language; classify the original text, no translation.
3. Gemini then writes a short summary in Ukrainian from aggregated stats + a sample of comments:
   4–6 bullet points ("what the audience thinks"), plus the top 5 viewer questions/requests.
4. **Baseline:** VADER (`vaderSentiment`) on the original text, same thresholds as the reference
   project (compound >= 0.05 positive, <= -0.05 negative, else neutral). This replicates last year's
   approach (github.com/YARA-SENPAI/cloud_computing_project) so we can show why an LLM API is better.

Model: read `GEMINI_MODEL` from `.env`, default to a current free-tier Flash model. First check which
models are actually available with `client.models.list()` and choose a current stable Flash model
(not preview, not Pro); tell the user which one you chose.

Robustness:
- Retry on Gemini 429 / 5xx with exponential backoff, respect the suggested retry delay;
  small pause between batches to stay under free-tier RPM.
- If Gemini returns a malformed or incomplete batch, retry that batch once, then mark missing items
  as `unknown` instead of crashing.
- Handle: invalid URL, comments disabled (403 `commentsDisabled`), video not found, quota exceeded
  (403 `quotaExceeded`), video with 0 comments — show a clear message in Ukrainian.
- Cache raw comments and classifications as JSON in `data/cache/` (key: video id + model + N)
  so re-runs during the demo are instant and do not burn quota.
- Measure and report: YouTube fetch time, Gemini time, number of Gemini requests, tokens used
  (from usage metadata), total time.

## Interface: Streamlit (`app.py`), UI text in Ukrainian
- Input field for URL, slider for number of comments (100–500), "Аналізувати" button.
- Video header (title, channel, views, comment count).
- Metrics row: total comments, % positive / neutral / negative, # questions, # toxic, timings.
- Charts: sentiment distribution (Gemini vs VADER side by side), top topics bar chart.
- Gemini summary block + top viewer questions.
- Toxic comments list (collapsible).
- Full table with filters by sentiment/topic; download CSV and JSON.
- Streamlit charts or plotly are fine; keep it clean.

## CLI
- `python cli.py analyze <url> --max 300` → prints summary, writes `reports/<video_id>.csv` and `.json`.

## Evaluation (for the presentation)
1. `python make_labeling_set.py <url1> <url2> <url3> --n 100 --seed 42`
   → `data/labeling/to_label.csv` with columns `id, video_id, text, label` (label empty),
   ~equal number of comments per video, skip empty/emoji-only duplicates.
   **Do not fill in labels yourself.** Tell the user to label each row `positive` / `neutral` /
   `negative` and save as `data/labeling/labeled.csv`, then stop and wait.
2. `python evaluate.py data/labeling/labeled.csv`
   → runs Gemini and VADER on the same labeled comments, outputs:
   - accuracy, macro-F1, per-class precision/recall/F1 for both methods;
   - confusion matrix PNGs (`results/confusion_gemini.png`, `results/confusion_vader.png`);
   - comparison bar chart `results/accuracy_f1.png`;
   - `results/metrics.json`, `results/metrics.md` (table ready for slides);
   - `results/disagreements.csv`: examples where VADER is wrong and Gemini is right (for slides);
   - Gemini time and number of requests.
   Use scikit-learn for metrics, matplotlib for PNGs.

## Demo videos
- Video 1 (chosen by the user): https://www.youtube.com/watch?v=w224tgfrTBY
  ("If I Was Starting My Style Over, I'd Buy These 5 Things", mindofgabe, English).
- Pick 2 more with `search.list` (100 units each, fine): one **Ukrainian-language** video with
  >= 500 comments, and one English video with **polarized / mixed reactions** (e.g. a controversial
  product or tech review). Save ids + titles in `data/demo_videos.json` and tell the user what you picked.

## Project structure
```
app.py                  # Streamlit UI
cli.py
make_labeling_set.py
evaluate.py
commentlens/
  __init__.py
  config.py             # loads .env, constants
  youtube_client.py     # extract_video_id, fetch_video_info, fetch_comments (pagination)
  gemini_classifier.py  # batching, schema, retries, summary
  vader_baseline.py
  analysis.py           # aggregation, stats, dataframe building
  cache.py
tests/                  # pytest, all external calls mocked, no network, no keys needed
requirements.txt
.env.example
.gitignore
README.md
```

## Tests (pytest)
At least ~10 tests, all offline with mocks:
`extract_video_id` (watch URL, youtu.be, shorts, raw 11-char id, invalid), pagination stops at N
and on missing `nextPageToken`, error mapping (commentsDisabled, quotaExceeded), batching splits
correctly, malformed Gemini batch → retry then `unknown`, VADER thresholds, aggregation/percentages,
metrics computation on a tiny fixed example.

## README.md (in Ukrainian)
- What the project does and where it is useful (creators, marketing, moderation, research).
- APIs used: YouTube Data API v3 (endpoints, quota cost), Gemini API (model, structured output,
  free-tier limits).
- Architecture diagram (mermaid): User → Streamlit → YouTube Data API → Gemini API → results/cache.
- Setup guide: create GCP project, enable YouTube Data API v3, create + restrict API key,
  Gemini key in AI Studio, `.env`, venv, `pip install -r requirements.txt`, run commands.
- Security notes (key restriction, `.env` not committed).
- Results section (fill after evaluation: metrics table + charts).

## Workflow — do it in this order, report briefly after each step
1. Scaffold the project, `requirements.txt`, `.gitignore`, `.env.example`. Install deps in `.venv`.
2. Implement modules + tests; run `pytest` until green.
3. Smoke test with real APIs: `python cli.py analyze https://www.youtube.com/watch?v=w224tgfrTBY --max 100`.
   Fix any issues.
4. Run `streamlit run app.py` and tell the user to open it and check.
5. Pick the 2 extra demo videos, generate `to_label.csv`, then STOP and ask the user to label it.
6. After the user labels: run `evaluate.py`, fill the README results section.
7. Git: `git init`, commit (verify no secrets), ask the user for the GitHub repo URL, push.
