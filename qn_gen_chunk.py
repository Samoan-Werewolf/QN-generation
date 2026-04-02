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

def generate_qa_for_chunk(chunk: dict, n: int, dry_run: bool) -> list[dict]:
    if dry_run:
        return [
            {
                "question_number": i,
                "question": f"[DRY RUN] Sample question {i} for '{chunk['title']}'",
            }
            for i in range(1, n + 1)
        ]

    prompt = HR_QA_PROMPT.format(
        section_title   = chunk["title"],
        section_content = chunk["content"][:3000],
        n               = n,
    )
    raw = call_azure(prompt)

    try:
        qa_list = parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON for section '{chunk['title']}': {e}\n\nRaw:\n{raw}")

    if not isinstance(qa_list, list) or len(qa_list) == 0:
        raise ValueError(f"Empty or invalid Q&A list for section '{chunk['title']}'.")

    return qa_list


def process_file(md_path: Path, n: int, dry_run: bool) -> dict:
    text   = md_path.read_text(encoding="utf-8")
    chunks = chunk_by_header(text)

    if not chunks:
        return None

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
        get_client()
        return jsonify({"ok": True, "message": "Azure OpenAI credentials loaded"})
    except RuntimeError as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/run", methods=["POST"])
def run():
    try:
        data    = request.get_json(silent=True) or {}
        dry_run = data.get("dry_run", False)
        n       = int(data.get("questions_per_chunk", QUESTIONS_PER_CHUNK))

        if not INPUT_DIR.exists():
            return jsonify({"ok": False, "message": f"Input folder not found: '{INPUT_DIR}'."}), 400

        md_files = list(INPUT_DIR.glob("*.md")) + list(INPUT_DIR.glob("*.markdown"))
        if not md_files:
            return jsonify({"ok": False, "message": f"No .md or .markdown files found in '{INPUT_DIR}'."}), 400

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        summary = []
        for md_path in sorted(md_files):
            output = process_file(md_path, n=n, dry_run=dry_run)

            if output is None:
                summary.append({"file": md_path.name, "status": "skipped", "reason": "No ## headers found"})
                continue

            out_path = OUTPUT_DIR / (md_path.stem + "_doc.json")
            out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

            summary.append({
                "file":            md_path.name,
                "status":          "ok",
                "output":          out_path.name,
                "total_sections":  output["total_sections"],
                "total_questions": output["total_questions"],
            })

        return jsonify({
            "ok":              True,
            "files_found":     len(md_files),
            "files_processed": sum(1 for s in summary if s["status"] == "ok"),
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

    return send_file(buf, as_attachment=True, download_name="hr_policy_qa_all.zip", mimetype="application/zip")


@app.route("/list")
def list_outputs():
    if not OUTPUT_DIR.exists():
        return jsonify({"ok": True, "files": []})
    files = [f.name for f in sorted(OUTPUT_DIR.glob("*.json"))]
    return jsonify({"ok": True, "output_folder": str(OUTPUT_DIR), "files": files})


if __name__ == "__main__":
    app.run(debug=True)