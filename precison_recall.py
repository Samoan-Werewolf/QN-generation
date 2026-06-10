"""
Claim-based Precision / Recall / F1 for chatbot answer evaluation.

Integrates with your existing evaluation script (Ollama-based LLM judge).

  - PRECISION : fraction of claims in the chatbot's answer that are
                supported by the knowledge document (faithfulness /
                hallucination check). Needs only question + answer + knowledge.
  - RECALL    : fraction of facts in the ground-truth answer that are
                covered by the chatbot's answer (completeness check).
                REQUIRES an 'expected_answer' column in your CSV.
                If missing, recall and F1 are reported as None.
  - F1        : harmonic mean of precision and recall.

HOW TO INTEGRATE
----------------
1. Place this file next to your main script and add:
       from precision_recall_metrics import compute_precision_recall_f1
2. In evaluate_csv(), read the optional ground truth column:
       expected = row.get("expected_answer", "").strip()
3. After scores = score_answer(...), add:
       pr = compute_precision_recall_f1(answer, knowledge_context,
                                        expected, call_ollama)
       results.append({..., **scores, **pr})
   (pass your existing call_ollama function in — no duplication.)
4. Add 'expected_answer' as a column in your answers CSV files
   if you want recall and F1. Your simulated-answer CSV from earlier
   can serve as the ground truth source.
"""

import json
import re

# ── PROMPTS ────────────────────────────────────────────

EXTRACT_CLAIMS_PROMPT = """
Extract every distinct factual claim from the text below.
A claim is a single, self-contained statement of fact.
Ignore greetings, hedges, and offers to help.

Text:
{text}

Respond ONLY with a valid JSON array of strings. No preamble, no markdown fences.
Example: ["New employees are eligible after 3 months", "Workplace injuries have no waiting period"]
""".strip()

VERIFY_CLAIM_PROMPT = """
You are verifying whether a claim is supported by a reference text.

Reference text:
---
{reference}
---

Claim:
{claim}

Is this claim directly supported by the reference text?
Answer ONLY with a JSON object, no preamble, no markdown fences:
{{"supported": true}} or {{"supported": false}}
""".strip()


# ── PARSING HELPERS ────────────────────────────────────

def _strip_fences(raw: str) -> str:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    return re.sub(r',\s*([}\]])', r'\1', clean)


def _parse_json(raw: str, default):
    try:
        return json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, IndexError):
        return default


# ── CORE STEPS ─────────────────────────────────────────

def extract_claims(text: str, call_llm) -> list[str]:
    """Split a text into individual factual claims using the LLM judge."""
    if not text:
        return []
    raw = call_llm(EXTRACT_CLAIMS_PROMPT.format(text=text))
    claims = _parse_json(raw, default=[])
    return [c for c in claims if isinstance(c, str) and c.strip()]


def claim_supported(claim: str, reference: str, call_llm) -> bool:
    """Check a single claim against a reference text. Defaults to False
    (unsupported) when the judge response is unparseable — conservative
    so hallucinations are not silently counted as supported."""
    raw = call_llm(VERIFY_CLAIM_PROMPT.format(reference=reference, claim=claim))
    result = _parse_json(raw, default={})
    return bool(result.get("supported", False))


# ── PUBLIC API ─────────────────────────────────────────

def compute_precision_recall_f1(
    answer: str,
    knowledge_context: str,
    expected_answer: str,
    call_llm,
) -> dict:
    """
    Returns a dict ready to merge into your results row:
        answer_precision : float 0-1, or None if answer has no claims
        answer_recall    : float 0-1, or None if no expected_answer given
        f1               : float 0-1, or None if either component is None
        claims_total     : claims found in the chatbot answer
        claims_supported : how many were supported by the knowledge doc
    """
    # ---- PRECISION: answer claims vs knowledge document ----
    answer_claims = extract_claims(answer, call_llm)
    precision = None
    supported = 0
    if answer_claims:
        supported = sum(
            claim_supported(c, knowledge_context, call_llm)
            for c in answer_claims
        )
        precision = round(supported / len(answer_claims), 3)

    # ---- RECALL: ground-truth claims vs chatbot answer ----
    recall = None
    if expected_answer:
        gt_claims = extract_claims(expected_answer, call_llm)
        if gt_claims:
            covered = sum(
                claim_supported(c, answer, call_llm)
                for c in gt_claims
            )
            recall = round(covered / len(gt_claims), 3)

    # ---- F1 ----
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 3)
    elif precision is not None and recall is not None:
        f1 = 0.0

    return {
        "answer_precision": precision,
        "answer_recall":    recall,
        "f1":               f1,
        "claims_total":     len(answer_claims),
        "claims_supported": supported,
    }