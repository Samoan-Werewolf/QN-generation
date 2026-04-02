from flask import Flask, render_template, request, jsonify, send_file, abort
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import re
import os
import time
import zipfile
import io
from pathlib import Path

load_dotenv()

app = Flask(__name__)

# ── CONFIG ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
INPUT_DIR  = BASE_DIR / "policies"
OUTPUT_DIR = BASE_DIR / "questions"

AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT")       # e.g. https://your-resource.openai.azure.com/
AZURE_API_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT     = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

QUESTIONS_PER_FILE  = 25
# ───────────────────────────────────────────────────────

HR_QA_PROMPT = """
You are an HR assistant helping employees understand the company's HR Policies.

Below is the full HR policy document:

---
{policy_content}
---

A new employee has just joined and wants to understand this policy.

Your task:
Generate exactly {n} clear, practical questions that an employee might commonly ask about this policy.

Cover a diverse range of topics across the policy (e.g. eligibility, claims, coverage limits,
dependents, exclusions).

Respond ONLY with a valid JSON array. No preamble, no explanation, no markdown fences.
The JSON must follow this exact structure:

[
  {{
    "question_number": 1,
    "question": "The employee's question here"
  }}
]
""".strip()


# ── AZURE OPENAI ────────────────────────────────────────

def get_client() -> AzureOpenAI:
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        raise RuntimeError("Azure credentials missing. Check your .env file.")
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )


def call_azure(prompt: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


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


# ── Q&A GENERATION ─────────────────────────────────────

def generate_questions(policy_content: str, n: int, dry_run: bool) -> list[dict]:
    if dry_run:
        return [
            {"question_number": i, "question": f"[DRY RUN] Sample question {i}"}
            for i in range(1, n + 1)
        ]

    prompt = HR_QA_PROMPT.format(policy_content=policy_content[:6000], n=n)
    raw    = call_azure(prompt)

    try:
        questions = parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\n\nRaw:\n{raw}")

    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Expected a JSON array of questions but got something else.")

    return questions


def process_file(md_path: Path, n: int, dry_run: bool) -> dict:
    policy_content = md_path.read_text(encoding="utf-8")
    questions      = generate_questions(policy_content, n=n, dry_run=dry_run)

    return {
        "policy_source":  md_path.name,
        "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_count": len(questions),
        "questions":      questions,
    }


# ── ROUTES ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        get_client()
        return jsonify({"ok": True, "message": "Azure OpenAI credentials loaded"})
    except RuntimeError as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/run", methods=["POST"])
def run():
    try:
        data    = request.get_json(silent=True) or {}
        dry_run = data.get("dry_run", False)
        n       = int(data.get("questions_per_file", QUESTIONS_PER_FILE))

        if not INPUT_DIR.exists():
            return jsonify({"ok": False, "message": f"Input folder not found: '{INPUT_DIR}'."}), 400

        md_files = sorted(INPUT_DIR.glob("*.md")) + sorted(INPUT_DIR.glob("*.markdown"))
        if not md_files:
            return jsonify({"ok": False, "message": f"No .md or .markdown files found in '{INPUT_DIR}'."}), 400

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        summary = []
        for md_path in md_files:
            output   = process_file(md_path, n=n, dry_run=dry_run)
            out_path = OUTPUT_DIR / (md_path.stem + "_doc.json")
            out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

            summary.append({
                "file":           md_path.name,
                "status":         "ok",
                "output":         out_path.name,
                "question_count": output["question_count"],
            })

        return jsonify({
            "ok":              True,
            "files_processed": len(summary),
            "output_folder":   str(OUTPUT_DIR),
            "summary":         summary,
        })

    except (ValueError, RuntimeError) as e:
        return jsonify({"ok": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    out_path = OUTPUT_DIR / filename
    if not out_path.exists() or out_path.suffix != ".json":
        abort(404)
    return send_file(out_path, as_attachment=True, mimetype="application/json")


@app.route("/download_all")
def download_all():
    json_files = list(OUTPUT_DIR.glob("*.json"))
    if not json_files:
        return "No JSON files generated yet. Please call /run first.", 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in json_files:
            zf.write(f, f.name)
    buf.seek(0)

    return send_file(buf, as_attachment=True, download_name="hr_policy_questions.zip", mimetype="application/zip")


@app.route("/list")
def list_outputs():
    if not OUTPUT_DIR.exists():
        return jsonify({"ok": True, "files": []})
    files = [f.name for f in sorted(OUTPUT_DIR.glob("*.json"))]
    return jsonify({"ok": True, "output_folder": str(OUTPUT_DIR), "files": files})


if __name__ == "__main__":
    app.run(debug=True)