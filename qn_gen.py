from flask import Flask, render_template, request, jsonify, send_file, abort
import requests
import json
import re
import time
import zipfile
import io
from pathlib import Path

app = Flask(__name__)

# ── CONFIG ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
INPUT_DIR  = BASE_DIR / "policies"       # folder of .md / .markdown files
OUTPUT_DIR = BASE_DIR / "questions"      # folder where JSONs are written

OLLAMA_URL          = "http://localhost:11434/api/generate"
MODEL_NAME          = "llama3"
OLLAMA_TIMEOUT      = 120
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


# ── OLLAMA ─────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2000},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"Error: Ollama returned status {resp.status_code}\n{resp.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot reach Ollama. Is it running? Try: ollama serve"
    except requests.exceptions.Timeout:
        return f"Error: Ollama timed out after {OLLAMA_TIMEOUT}s."
    except Exception as e:
        return f"Error: {e}"


def parse_json_response(raw: str) -> list[dict]:
    """Strip markdown fences, fix common model JSON typos, then parse."""
    clean = raw.strip()

    # Strip markdown fences
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    # Remove invalid backslash escapes (model over-escapes $, @, +, %, #, etc.)
    clean = re.sub(r'\\([^\"\\\/bfnrtu])', r'\1', clean)

    # Fix stray quotes after numbers e.g. "question_number": 4"
    clean = re.sub(r':\s*(\d+)"', r': \1', clean)

    # Fix trailing commas before ] or }
    clean = re.sub(r',\s*([}\]])', r'\1', clean)

    # If model truncated the JSON, close any open structures
    open_braces = clean.count('{') - clean.count('}')
    if open_braces > 0:
        clean = clean.rstrip(',\n ') + ('}' * open_braces)
    if not clean.rstrip().endswith(']'):
        clean = clean.rstrip() + ']'

    return json.loads(clean)


# ── Q&A GENERATION ─────────────────────────────────────

def generate_questions(policy_content: str, n: int, dry_run: bool) -> list[dict]:
    """Send the full policy to Ollama and return n questions as a list."""
    if dry_run:
        return [
            {"question_number": i, "question": f"[DRY RUN] Sample question {i}"}
            for i in range(1, n + 1)
        ]

    prompt = HR_QA_PROMPT.format(policy_content=policy_content[:6000], n=n)
    raw    = call_ollama(prompt)

    if raw.startswith("Error:"):
        raise RuntimeError(raw)

    try:
        questions = parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\n\nRaw:\n{raw}")

    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Expected a JSON array of questions but got something else.")

    return questions


def process_file(md_path: Path, n: int, dry_run: bool) -> dict:
    """Read one markdown file and generate questions from the full content."""
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
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            return jsonify({"ok": True, "message": "Ollama is running"})
        return jsonify({"ok": False, "message": f"Ollama returned {resp.status_code}"}), 500
    except Exception:
        return jsonify({"ok": False, "message": "Ollama is not running"}), 500


@app.route("/run", methods=["POST"])
def run():
    try:
        data    = request.get_json(silent=True) or {}
        dry_run = data.get("dry_run", False)
        n       = int(data.get("questions_per_file", QUESTIONS_PER_FILE))

        # Validate input folder
        if not INPUT_DIR.exists():
            return jsonify({
                "ok": False,
                "message": f"Input folder not found: '{INPUT_DIR}'. Create it and add markdown files."
            }), 400

        md_files = sorted(INPUT_DIR.glob("*.md")) + sorted(INPUT_DIR.glob("*.markdown"))
        if not md_files:
            return jsonify({
                "ok": False,
                "message": f"No .md or .markdown files found in '{INPUT_DIR}'."
            }), 400

        # Create output folder if it doesn't exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Process each file
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
    """Download a specific output JSON by filename e.g. /download/HR_POLICY.json"""
    out_path = OUTPUT_DIR / filename
    if not out_path.exists() or out_path.suffix != ".json":
        abort(404)
    return send_file(out_path, as_attachment=True, mimetype="application/json")


@app.route("/download_all")
def download_all():
    """Download all output JSONs bundled as a zip file."""
    json_files = list(OUTPUT_DIR.glob("*.json"))
    if not json_files:
        return "No JSON files generated yet. Please call /run first.", 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in json_files:
            zf.write(f, f.name)
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name="hr_policy_questions.zip",
        mimetype="application/zip",
    )


@app.route("/list")
def list_outputs():
    """List all generated JSON files in the output folder."""
    if not OUTPUT_DIR.exists():
        return jsonify({"ok": True, "files": []})
    files = [f.name for f in sorted(OUTPUT_DIR.glob("*.json"))]
    return jsonify({"ok": True, "output_folder": str(OUTPUT_DIR), "files": files})


if __name__ == "__main__":
    app.run(debug=True)