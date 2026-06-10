import csv
import json
import re
import time
import requests
from pathlib import Path
from precison_recall import compute_precision_recall_f1

# ── CONFIG ─────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL_NAME     = "llama3.2"
OLLAMA_TIMEOUT = 120
MAX_RETRIES    = 3   # number of attempts before giving up
RETRY_DELAY    = 2   # seconds to wait between retries

BASE_DIR       = Path(__file__).resolve().parent
ANSWERS_DIR    = BASE_DIR / "answers"
SCORED_DIR     = BASE_DIR / "scored"
KNOWLEDGE_DIR  = BASE_DIR / "knowledge"   # same folder used by qn_gen_personal.py

# File types to load from KNOWLEDGE_DIR
KNOWLEDGE_EXTENSIONS = [".md", ".markdown", ".txt", ".csv", ".json"]

# Max characters of knowledge context sent to the model per scoring call
KNOWLEDGE_MAX_CHARS = 6000

# ── METRICS ────────────────────────────────────────────
# Add, edit, or remove metrics here.
# Each entry needs: name, description, scale.
METRICS = [
    {
        "name":        "relevance",
        "description": "How relevant is the answer to the question?",
        "scale":       "1 to 5 (1 = completely irrelevant, 5 = highly relevant)",
    },
    {
        "name":        "completeness",
        "description": "How completely does the answer address the question?",
        "scale":       "1 to 5 (1 = very incomplete, 5 = fully addressed)",
    },
    {
        "name":        "clarity",
        "description": "How clear and easy to understand is the answer?",
        "scale":       "1 to 5 (1 = very confusing, 5 = very clear)",
    },
    # {
    #     "name":        "accuracy",
    #     "description": "How factually accurate is the answer based on the knowledge document?",
    #     "scale":       "1 to 5 (1 = contradicts or ignores the document, 5 = fully grounded in the document)",
    # },
]

# ── LABELLING ──────────────────────────────────────────
# Edit this function to define your own pass/fail logic.
# avg_score is the mean of all metric scores (out of 5).
PASS_THRESHOLD = 3.0

def label(avg_score: float) -> str:
    return "pass" if avg_score >= PASS_THRESHOLD else "fail"
# ───────────────────────────────────────────────────────


SCORE_PROMPT = """
You are an objective evaluator assessing the quality of a chatbot's answer.

The chatbot is based on the following knowledge document:

---
{knowledge_context}
---

Question asked:
{question}

Chatbot Answer:
{answer}

Evaluate the answer on each of the following metrics and assign an integer score:

{metrics_block}

Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown fences.
The JSON must follow this exact structure:

{{
{score_fields}
}}
""".strip()


def build_prompt(question: str, answer: str, knowledge_context: str) -> str:
    metrics_block = "\n".join(
        f"- {m['name']}: {m['description']} Score: {m['scale']}"
        for m in METRICS
    )
    score_fields = ",\n".join(
        f'  "{m["name"]}": <integer score>' for m in METRICS
    )
    return SCORE_PROMPT.format(
        knowledge_context=knowledge_context,
        question=question,
        answer=answer,
        metrics_block=metrics_block,
        score_fields=score_fields,
    )


def read_knowledge_dir() -> str:
    if not KNOWLEDGE_DIR.exists():
        return ""

    parts = []
    for path in sorted(KNOWLEDGE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in KNOWLEDGE_EXTENSIONS:
            continue
        try:
            if path.suffix.lower() in (".md", ".markdown", ".txt"):
                parts.append(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() == ".csv":
                rows = []
                with path.open(encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
                parts.append("\n".join(rows))
            elif path.suffix.lower() == ".json":
                parts.append(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        except Exception:
            continue

    combined = "\n\n".join(parts)
    return combined[:KNOWLEDGE_MAX_CHARS]


def call_ollama(prompt: str) -> str:
    payload = {
        "model":   MODEL_NAME,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 500},
    }
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            last_error = f"Error: Ollama returned status {resp.status_code}"
        except requests.exceptions.ConnectionError:
            return "Error: Cannot reach Ollama. Is it running? Try: ollama serve"
        except requests.exceptions.Timeout:
            last_error = f"Error: Ollama timed out after {OLLAMA_TIMEOUT}s."
        except Exception as e:
            last_error = f"Error: {e}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return last_error


def parse_scores(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    clean = re.sub(r',\s*([}\]])', r'\1', clean)
    return json.loads(clean)


def score_answer(question: str, answer: str, knowledge_context: str) -> dict:
    prompt = build_prompt(question, answer, knowledge_context)
    raw    = call_ollama(prompt)

    if raw.startswith("Error:"):
        raise RuntimeError(raw)

    try:
        scores = parse_scores(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\n\nRaw:\n{raw}")

    metric_names = [m["name"] for m in METRICS]
    values       = [float(scores.get(name, 0)) for name in metric_names]
    avg          = round(sum(values) / len(values), 2) if values else 0.0

    return (
        {name: scores.get(name) for name in metric_names}
        | {"overall_score": avg, "label": label(avg)}
    )


def evaluate_csv(input_path: Path, knowledge_context: str) -> list[dict]:
    results = []
    with input_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question        = row.get("question",         "").strip()
            answer          = row.get("answer",           "").strip()
            qnum            = row.get("question_number",  "")
            expected_answer = row.get("expected_answer",  "").strip()

            if not question or not answer:
                continue

            print(f"  Scoring Q{qnum}: {question[:70]}...")
            scores = score_answer(question, answer, knowledge_context)
            pr     = compute_precision_recall_f1(answer, knowledge_context, expected_answer, call_ollama)

            results.append({
                "question_number": qnum,
                "question":        question,
                "answer":          answer,
                **scores,
                **pr,
            })

    return results


def write_output(results: list[dict], output_path: Path) -> None:
    if not results:
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    total  = len(results)
    passed = sum(1 for r in results if r["label"] == "pass")
    failed = total - passed
    print(f"\n{'─' * 42}")
    print(f"  Total evaluated : {total}")
    print(f"  Pass            : {passed}  ({passed / total * 100:.1f}%)")
    print(f"  Fail            : {failed}  ({failed / total * 100:.1f}%)")
    print(f"{'─' * 42}\n")


def main():
    if not ANSWERS_DIR.exists():
        print(f"Error: answers folder not found at '{ANSWERS_DIR}'. Create it and add CSV files.")
        return

    csv_files = sorted(ANSWERS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{ANSWERS_DIR}'.")
        return

    SCORED_DIR.mkdir(parents=True, exist_ok=True)

    knowledge_context = read_knowledge_dir()
    if knowledge_context:
        print(f"\nKnowledge : {KNOWLEDGE_DIR} ({len(knowledge_context)} chars loaded)")
    else:
        print(f"\nKnowledge : none found in '{KNOWLEDGE_DIR}' — scoring without document context")

    print(f"Model     : {MODEL_NAME}")
    print(f"Metrics   : {', '.join(m['name'] for m in METRICS)}")
    print(f"Threshold : {PASS_THRESHOLD} / 5.0")

    for csv_path in csv_files:
        print(f"\nEvaluating: {csv_path.name}")
        results = evaluate_csv(csv_path, knowledge_context)

        if not results:
            print(f"  Skipped — no valid 'question' and 'answer' columns found.")
            continue

        output_path = SCORED_DIR / (csv_path.stem + "_scored.csv")
        write_output(results, output_path)
        print(f"  Scored output written to: {output_path}")
        print_summary(results)


if __name__ == "__main__":
    main()
