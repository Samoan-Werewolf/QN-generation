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
BASE_DIR    = Path(__file__).resolve().parent
INPUT_DIR   = BASE_DIR / "policies"          # folder of .md / .markdown files
OUTPUT_DIR  = BASE_DIR / "questions"       # folder where JSONs are written

OLLAMA_URL          = "http://localhost:11434/api/generate"
MODEL_NAME          = "llama3"
OLLAMA_TIMEOUT      = 120
QUESTIONS_PER_CHUNK = 5
# ───────────────────────────────────────────────────────

HR_QA_PROMPT = """
You are an HR assistant helping employees understand the company's HR Policies.

Below is ONE section of the HR policy document titled "{section_title}":

---
{section_content}
---

A new employee has just joined and wants to understand this section.

Your task:
Generate exactly {n} clear, practical questions that an employee might commonly ask about THIS section
Do not rephrase questions that have already been asked. Do not repeat questions.
Respond ONLY with a valid JSON array. No preamble, no explanation, no markdown fences.
The JSON must follow this exact structure:

[
  {{
    "question_number": 1,
    "question": "The employee's question here"
  }}
]
""".strip()


# ── CHUNKING ───────────────────────────────────────────

def chunk_by_header(md_text: str) -> list[dict]:
    """Split markdown into chunks at every ## header."""
    header_pattern = re.compile(r"^(#{2})\s+(.+)", re.MULTILINE)
    matches = list(header_pattern.finditer(md_text))

    chunks = []
    for i, match in enumerate(matches):
        title   = match.group(2).strip()
        start   = match.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()
        if len(content) >= 30:
            chunks.append({"title": title, "content": content})

    return chunks


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

def generate_qa_for_chunk(chunk: dict, n: int, dry_run: bool) -> list[dict]:
    """Generate n Q&A pairs for a single markdown chunk."""
    if dry_run:
        return [
            {
                "question_number": i,
                "question": f"[DRY RUN] Sample question {i} for '{chunk['title']}'",
                "answer":   "[DRY RUN] Ollama was not called.",
            }
            for i in range(1, n + 1)
        ]

    prompt = HR_QA_PROMPT.format(
        section_title   = chunk["title"],
        section_content = chunk["content"][:3000],
        n               = n,
    )
    raw = call_ollama(prompt)

    if raw.startswith("Error:"):
        raise RuntimeError(raw)

    try:
        qa_list = parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON for section '{chunk['title']}': {e}\n\nRaw:\n{raw}"
        )

    if not isinstance(qa_list, list) or len(qa_list) == 0:
        raise ValueError(f"Empty or invalid Q&A list for section '{chunk['title']}'.")

    return qa_list


def process_file(md_path: Path, n: int, dry_run: bool) -> dict:
    """Chunk one markdown file and generate Q&A for each chunk."""
    text   = md_path.read_text(encoding="utf-8")
    chunks = chunk_by_header(text)

    if not chunks:
        return None  # skip files with no ## headers

    sections = []
    for chunk in chunks:
        qa_list = generate_qa_for_chunk(chunk, n=n, dry_run=dry_run)
        sections.append({
            "section":        chunk["title"],
            "question_count": len(qa_list),
            "qa_pairs":       qa_list,
        })
        time.sleep(0.3)

    return {
        "policy_source":   str(md_path.name),
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_sections":  len(sections),
        "total_questions": sum(s["question_count"] for s in sections),
        "sections":        sections,
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
        n       = int(data.get("questions_per_chunk", QUESTIONS_PER_CHUNK))

        # Validate input folder
        if not INPUT_DIR.exists():
            return jsonify({
                "ok": False,
                "message": f"Input folder not found: '{INPUT_DIR}'. Create it and add markdown files."
            }), 400

        md_files = list(INPUT_DIR.glob("*.md")) + list(INPUT_DIR.glob("*.markdown"))
        if not md_files:
            return jsonify({
                "ok": False,
                "message": f"No .md or .markdown files found in '{INPUT_DIR}'."
            }), 400

        # Create output folder if it doesn't exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Process each file
        summary = []
        for md_path in sorted(md_files):
            output = process_file(md_path, n=n, dry_run=dry_run)

            if output is None:
                summary.append({
                    "file": md_path.name,
                    "status": "skipped",
                    "reason": "No ## headers found",
                })
                continue

            # Write JSON with same stem as the markdown file
            out_path = OUTPUT_DIR / (md_path.stem + ".json")
            out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

            summary.append({
                "file":            md_path.name,
                "status":          "ok",
                "output":          out_path.name,
                "total_sections":  output["total_sections"],
                "total_questions": output["total_questions"],
            })

        return jsonify({
            "ok":           True,
            "files_found":  len(md_files),
            "files_processed": sum(1 for s in summary if s["status"] == "ok"),
            "output_folder": str(OUTPUT_DIR),
            "summary":      summary,
        })

    except (ValueError, RuntimeError) as e:
        return jsonify({"ok": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    """Download a specific output JSON by filename."""
    out_path = OUTPUT_DIR / filename
    if not out_path.exists() or out_path.suffix != ".json":
        abort(404)
    return send_file(out_path, as_attachment=True, mimetype="application/json")


@app.route("/download_all")
def download_all():
    """Download all output JSONs as a zip file."""
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
        download_name="hr_policy_qa_all.zip",
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