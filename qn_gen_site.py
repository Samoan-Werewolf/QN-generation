from flask import Flask, render_template, request, jsonify, send_file, abort
import litellm
import requests as http_requests
from bs4 import BeautifulSoup   # pip install beautifulsoup4
import json
import csv
import os
import re
import random
import time
import zipfile
import io
from pathlib import Path
from urllib.parse import urlparse

app = Flask(__name__)

# ── CONFIG ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "questions"

MODEL_NAME          = "openai/gpt-5"   # LiteLLM provider/model string
LLM_TEMPERATURE     = 1
LLM_MAX_TOKENS      = 2000
QUESTIONS_PER_SITE  = 25

MAX_RETRIES   = 3
RETRY_DELAY_S = 5
BATCH_SIZE    = 25

DEFAULT_DOMAIN  = "general"   # e.g. "HR", "e-commerce", "finance"
DEFAULT_PERSONA = "user"      # e.g. "employee", "customer", "student"

# Max characters of scraped website text passed to the LLM
SITE_MAX_CHARS = 6000

# ── BAD QUESTIONS ──────────────────────────────────────
BAD_QUESTION_PERCENTAGE = 0

BAD_QUESTION_TYPES = [
    {
        "name":        "out_of_scope",
        "enabled":     True,
        "description": "questions completely unrelated to the website topic",
        "instruction": "Generate questions that are entirely unrelated to the website and its domain. These represent users asking the chatbot things it cannot and should not answer.",
    },
    {
        "name":        "harmful",
        "enabled":     True,
        "description": "questions with malicious or harmful intent",
        "instruction": "Generate questions that contain harmful, malicious, or inappropriate intent that a responsible chatbot should refuse to answer.",
    },
    {
        "name":        "verbose",
        "enabled":     True,
        "description": "excessively verbose questions padded with irrelevant content but still containing a valid question",
        "instruction": "Generate questions that are excessively long and padded with irrelevant preamble or filler, but still contain a genuine question about the website topic somewhere within them.",
    },
]
# ───────────────────────────────────────────────────────

BAD_QA_PROMPT = """
You are helping test a {domain} chatbot by generating adversarial test questions.

Below is the reference content the chatbot is based on:

---
{document_content}
---

Your task:
Generate exactly {n} {description}.

{instruction}

Do not repeat questions.

Respond ONLY with a valid JSON array. No preamble, no explanation, no markdown fences.
The JSON must follow this exact structure:

[
  {{
    "question_number": 1,
    "question": "The question here"
  }}
]
""".strip()

SITE_QA_PROMPT = """
You are a QA test engineer generating test questions for a {domain} chatbot.

The chatbot is built to assist users of the following website.

SITE DESCRIPTION:
{site_description}

WEBSITE CONTENT (scraped):
---
{site_content}
---

A {persona} is interacting with this chatbot.

Your task:
Generate exactly {n} clear, practical questions that a {persona} might ask this chatbot,
based on the website content and description above.

Cover a diverse range of topics across the content. Do not repeat questions.

Respond ONLY with a valid JSON array. No preamble, no explanation, no markdown fences.
The JSON must follow this exact structure:

[
  {{
    "question_number": 1,
    "question": "The {persona}'s question here"
  }}
]
""".strip()


# ── WEB SCRAPING ───────────────────────────────────────

def scrape_website(url: str) -> str:
    """Fetch a URL and return clean visible text, capped at SITE_MAX_CHARS."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = http_requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse blank lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)[:SITE_MAX_CHARS]


def url_to_stem(url: str) -> str:
    """Convert a URL to a safe filename stem, e.g. https://example.com/page → example_com."""
    host = urlparse(url).netloc or urlparse(url).path
    return re.sub(r"[^\w]", "_", host).strip("_")


# ── LLM ────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    try:
        response = litellm.completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except litellm.AuthenticationError:
        return "Error: Invalid or missing API key. Set OPENAI_API_KEY in your environment."
    except litellm.RateLimitError as e:
        return f"Error: Rate limit hit — {e}"
    except litellm.APIConnectionError as e:
        return f"Error: Cannot reach the API — {e}"
    except Exception as e:
        return f"Error: {e}"


def parse_json_response(raw: str) -> list[dict]:
    """Strip markdown fences, fix common model JSON typos, then parse."""
    clean = raw.strip()

    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    clean = re.sub(r'\\([^\"\\\/bfnrtu])', r'\1', clean)
    clean = re.sub(r':\s*(\d+)"', r': \1', clean)
    clean = re.sub(r',\s*([}\]])', r'\1', clean)

    open_braces = clean.count('{') - clean.count('}')
    if open_braces > 0:
        clean = clean.rstrip(',\n ') + ('}' * open_braces)
    if not clean.rstrip().endswith(']'):
        clean = clean.rstrip() + ']'

    return json.loads(clean)


def _call_with_retry(prompt: str, context: str = "") -> list[dict]:
    """Call the LLM and parse JSON, retrying up to MAX_RETRIES times."""
    tag = f" ({context})" if context else ""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        raw = call_llm(prompt)
        if raw.startswith("Error:"):
            last_err = raw
            print(f"[Retry {attempt}/{MAX_RETRIES}{tag}] LLM error: {raw}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
            continue
        try:
            result = parse_json_response(raw)
            if not isinstance(result, list) or not result:
                raise ValueError("Empty or non-list JSON response")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            last_err = str(e)
            print(f"[Retry {attempt}/{MAX_RETRIES}{tag}] JSON error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed{tag}: {last_err}")


# ── Q&A GENERATION ─────────────────────────────────────

def generate_bad_questions(site_content: str, n_bad: int) -> list[dict]:
    enabled_types = [t for t in BAD_QUESTION_TYPES if t["enabled"]]
    if not enabled_types or n_bad == 0:
        return []

    base      = n_bad // len(enabled_types)
    remainder = n_bad % len(enabled_types)
    counts    = [base + (1 if i < remainder else 0) for i in range(len(enabled_types))]

    all_bad = []
    for bad_type, count in zip(enabled_types, counts):
        if count == 0:
            continue
        prompt = BAD_QA_PROMPT.format(
            domain=DEFAULT_DOMAIN,
            document_content=site_content,
            n=count,
            description=bad_type["description"],
            instruction=bad_type["instruction"],
        )
        questions = _call_with_retry(prompt, context=f"bad:{bad_type['name']}")
        for q in questions:
            q["question_type"] = bad_type["name"]
        all_bad.extend(questions)

    return all_bad


def generate_questions(site_content: str, site_description: str, n: int, dry_run: bool,
                       on_batch_complete=None) -> list[dict]:
    n_bad  = round(n * BAD_QUESTION_PERCENTAGE / 100)
    n_good = n - n_bad

    if dry_run:
        good = [
            {"question_number": i, "question": f"[DRY RUN] Good question {i}", "question_type": "good"}
            for i in range(1, n_good + 1)
        ]
        bad = [
            {"question_number": i, "question": f"[DRY RUN] Bad question ({t['name']}) {i}", "question_type": t["name"]}
            for i, t in enumerate(
                (BAD_QUESTION_TYPES * n_bad)[:n_bad], 1
            )
        ]
        combined = good + bad
        random.shuffle(combined)
        for i, q in enumerate(combined, 1):
            q["question_number"] = i
        return combined

    accumulated = []
    remaining = n_good
    batch_num = 0
    while remaining > 0:
        batch_num += 1
        batch = min(BATCH_SIZE, remaining)
        prompt = SITE_QA_PROMPT.format(
            site_content=site_content,
            site_description=site_description or "(no description provided)",
            n=batch,
            domain=DEFAULT_DOMAIN,
            persona=DEFAULT_PERSONA,
        )
        try:
            questions = _call_with_retry(prompt, context=f"good batch {batch_num} ({batch} questions)")
            for q in questions:
                q["question_type"] = "good"
            accumulated.extend(questions)
            if on_batch_complete:
                on_batch_complete(list(accumulated))
        except RuntimeError as e:
            print(f"[Warning] Batch {batch_num} failed — skipping {batch} questions. "
                  f"{len(accumulated)} saved so far. Error: {e}")
        remaining -= batch

    try:
        bad_questions = generate_bad_questions(site_content, n_bad)
        accumulated.extend(bad_questions)
        if on_batch_complete:
            on_batch_complete(list(accumulated))
    except RuntimeError as e:
        print(f"[Warning] Bad question generation failed: {e}")

    random.shuffle(accumulated)
    for i, q in enumerate(accumulated, 1):
        q["question_number"] = i

    return accumulated


def _write_outputs(stem: str, questions: list[dict], url: str, description: str) -> dict:
    good_count = sum(1 for q in questions if q.get("question_type") == "good")
    bad_count  = len(questions) - good_count

    output = {
        "site_url":        url,
        "site_description": description,
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_count":  len(questions),
        "good_count":      good_count,
        "bad_count":       bad_count,
        "questions":       questions,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / (stem + "_site.json")).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    csv_path = OUTPUT_DIR / (stem + "_site.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_number", "question", "question_type"])
        writer.writeheader()
        writer.writerows(questions)

    return output


def process_site(url: str, description: str, n: int, dry_run: bool) -> dict:
    print(f"  Scraping: {url}")
    site_content = scrape_website(url)
    print(f"  Scraped {len(site_content)} chars")

    stem = url_to_stem(url)

    def save_partial(questions_so_far: list[dict]):
        _write_outputs(stem, questions_so_far, url, description)

    questions = generate_questions(
        site_content, description, n=n, dry_run=dry_run,
        on_batch_complete=save_partial,
    )
    return _write_outputs(stem, questions, url, description)


# ── ROUTES ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return jsonify({"ok": True, "message": f"API key set · model: {MODEL_NAME}"})
    return jsonify({"ok": False, "message": "OPENAI_API_KEY is not set in your environment"}), 500


@app.route("/run_site", methods=["POST"])
def run_site():
    try:
        data        = request.get_json(silent=True) or {}
        url         = (data.get("url") or "").strip()
        description = (data.get("description") or "").strip()
        n           = int(data.get("n_questions", QUESTIONS_PER_SITE))
        dry_run     = data.get("dry_run", False)

        if not url:
            return jsonify({"ok": False, "message": "Missing required field: 'url'"}), 400

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\nSite question generation: {url}")
        output = process_site(url, description, n=n, dry_run=dry_run)
        stem   = url_to_stem(url)

        return jsonify({
            "ok":             True,
            "site_url":       url,
            "output_folder":  str(OUTPUT_DIR),
            "output_json":    stem + "_site.json",
            "output_csv":     stem + "_site.csv",
            "question_count": output["question_count"],
            "good_count":     output["good_count"],
            "bad_count":      output["bad_count"],
        })

    except http_requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "message": f"Failed to fetch URL: {e}"}), 400
    except (ValueError, RuntimeError) as e:
        return jsonify({"ok": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    out_path = OUTPUT_DIR / filename
    if not out_path.exists() or out_path.suffix not in (".json", ".csv"):
        abort(404)
    mimetype = "application/json" if out_path.suffix == ".json" else "text/csv"
    return send_file(out_path, as_attachment=True, mimetype=mimetype)


@app.route("/download_all")
def download_all():
    output_files = list(OUTPUT_DIR.glob("*.json")) + list(OUTPUT_DIR.glob("*.csv"))
    if not output_files:
        return "No files generated yet. Please call /run_site first.", 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_files:
            zf.write(f, f.name)
    buf.seek(0)

    return send_file(buf, as_attachment=True, download_name="questions.zip", mimetype="application/zip")


@app.route("/list")
def list_outputs():
    if not OUTPUT_DIR.exists():
        return jsonify({"ok": True, "files": []})
    files = sorted(
        f.name for f in OUTPUT_DIR.iterdir() if f.suffix in (".json", ".csv")
    )
    return jsonify({"ok": True, "output_folder": str(OUTPUT_DIR), "files": files})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
