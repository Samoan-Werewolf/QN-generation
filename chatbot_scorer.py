import csv
import json
import re
import requests
import argparse
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL_NAME     = "llama3.2"
OLLAMA_TIMEOUT = 120

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

Question:
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


def build_prompt(question: str, answer: str) -> str:
    metrics_block = "\n".join(
        f"- {m['name']}: {m['description']} Score: {m['scale']}"
        for m in METRICS
    )
    score_fields = ",\n".join(
        f'  "{m["name"]}": <integer score>' for m in METRICS
    )
    return SCORE_PROMPT.format(
        question=question,
        answer=answer,
        metrics_block=metrics_block,
        score_fields=score_fields,
    )


def call_ollama(prompt: str) -> str:
    payload = {
        "model":   MODEL_NAME,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 500},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"Error: Ollama returned status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot reach Ollama. Is it running? Try: ollama serve"
    except requests.exceptions.Timeout:
        return f"Error: Ollama timed out after {OLLAMA_TIMEOUT}s."
    except Exception as e:
        return f"Error: {e}"


def parse_scores(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    clean = re.sub(r',\s*([}\]])', r'\1', clean)
    return json.loads(clean)


def score_answer(question: str, answer: str) -> dict:
    prompt = build_prompt(question, answer)
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


def evaluate_csv(input_path: Path) -> list[dict]:
    results = []
    with input_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "").strip()
            answer   = row.get("answer",   "").strip()
            qnum     = row.get("question_number", "")

            if not question or not answer:
                continue

            print(f"  Scoring Q{qnum}: {question[:70]}...")
            scores = score_answer(question, answer)

            results.append({
                "question_number": qnum,
                "question":        question,
                "answer":          answer,
                **scores,
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
    parser = argparse.ArgumentParser(
        description="Score chatbot CSV output using Llama 3.2."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input CSV file — must have 'question' and 'answer' columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>_scored.csv).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: '{args.input}' not found.")
        return

    output_path = args.output or args.input.with_stem(args.input.stem + "_scored")

    print(f"\nEvaluating: {args.input}")
    print(f"Model:      {MODEL_NAME}")
    print(f"Metrics:    {', '.join(m['name'] for m in METRICS)}")
    print(f"Threshold:  {PASS_THRESHOLD} / 5.0\n")

    results = evaluate_csv(args.input)

    if not results:
        print("No valid rows found. Make sure the CSV has 'question' and 'answer' columns.")
        return

    write_output(results, output_path)
    print(f"\nScored output written to: {output_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
