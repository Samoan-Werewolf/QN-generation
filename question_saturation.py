"""
QUESTION SATURATION FRAMEWORK
=============================================================================
Finds the "sweet spot" number of test questions for a policy document by
measuring how much NEW coverage each additional question adds, then locating
the elbow point where adding more questions stops paying off.

CORE IDEA
---------
Each question and each chunk of the policy is turned into a vector. As you
grow the question set (1, 2, 3, ... N), two things are tracked:

  1. COVERAGE   : fraction of document chunks "covered" by at least one
                  question (a chunk is covered if some question is semantically
                  close enough to it). Rises fast, then plateaus.
  2. REDUNDANCY : how similar each new question is to the questions already
                  in the set. Low at first, climbs as you run out of new
                  things to ask.

The COVERAGE curve saturates (flattens). The ELBOW of that curve is your
sweet spot: the point of diminishing returns.

ORDER SENSITIVITY
-----------------
A single question ordering gives a misleading curve (the elbow depends on
which questions happen to come first). The framework therefore runs many
random shuffles and averages the curves, plotting mean +/- standard deviation.

EMBEDDING BACKENDS (pluggable)
------------------------------
  - "ollama"  : semantic embeddings via your local Ollama (RECOMMENDED for
                production - matches the richer semantics Copilot uses).
                Requires:  ollama pull nomic-embed-text
  - "tfidf"   : lexical vectors, no model/network needed. Good for a quick
                run or where Ollama isn't available. Less semantic - treats
                "annual leave" and "vacation" as unrelated.

Switch via EMBED_BACKEND below.
"""

import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ─────────────────────────────────────────────
EMBED_BACKEND      = "tfidf"        # "ollama" or "tfidf"
OLLAMA_EMBED_URL   = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# A chunk counts as "covered" if a question's cosine similarity to it is at
# least this value. Calibrate per backend (TF-IDF sims run lower than
# embedding sims). Set to None to auto-calibrate from the similarity matrix.
COVERAGE_THRESHOLD = None

N_SHUFFLES = 30      # random orderings to average over (robustness)
RANDOM_SEED = 42
# ───────────────────────────────────────────────────────


# ── DOCUMENT CHUNKING ──────────────────────────────────
def chunk_markdown_by_heading(md_text: str) -> list[str]:
    """Split a markdown policy into chunks at each ### subsection heading.
    For a structured policy this mirrors the semantic units a reader (and a
    RAG retriever) would treat as distinct - better than fixed-token chunks
    for a small, well-structured document."""
    lines = md_text.splitlines()
    chunks, current = [], []
    for line in lines:
        # Start a new chunk at level-3+ headings (### or deeper)
        if re.match(r'^#{3,}\s', line):
            if current and any(c.strip() for c in current):
                chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current and any(c.strip() for c in current):
        chunks.append("\n".join(current).strip())
    # Drop chunks that are just a heading with no body
    return [c for c in chunks if len(c.split()) > 3]


# ── EMBEDDING BACKENDS ─────────────────────────────────
def embed_tfidf(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    return vec.fit_transform(texts).toarray()


def embed_ollama(texts: list[str]) -> np.ndarray:
    vectors = []
    for t in texts:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
            timeout=120,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embedding"])
    return np.array(vectors)


def embed(texts: list[str]) -> np.ndarray:
    if EMBED_BACKEND == "ollama":
        return embed_ollama(texts)
    return embed_tfidf(texts)


# ── SATURATION ANALYSIS ────────────────────────────────
def build_similarity_matrix(questions: list[str], chunks: list[str]):
    """Return (Q x C) cosine similarity matrix and the (Q x Q) question matrix.
    Both are embedded in one shared space so similarities are comparable."""
    all_vecs = embed(questions + chunks)
    q_vecs = all_vecs[: len(questions)]
    c_vecs = all_vecs[len(questions):]
    q_to_c = cosine_similarity(q_vecs, c_vecs)   # questions x chunks
    q_to_q = cosine_similarity(q_vecs, q_vecs)   # questions x questions
    return q_to_c, q_to_q


def auto_threshold(q_to_c: np.ndarray) -> float:
    """Pick a coverage threshold from the data: the median of each chunk's
    best-question similarity. Transparent and backend-agnostic."""
    best_per_chunk = q_to_c.max(axis=0)
    return float(np.median(best_per_chunk))


def coverage_curve(order: list[int], q_to_c: np.ndarray, threshold: float):
    """For one question ordering, return cumulative coverage at each k."""
    n_chunks = q_to_c.shape[1]
    covered = np.zeros(n_chunks, dtype=bool)
    curve = []
    for q_idx in order:
        covered |= (q_to_c[q_idx] >= threshold)
        curve.append(covered.sum() / n_chunks)
    return curve


def redundancy_curve(order: list[int], q_to_q: np.ndarray):
    """Redundancy of question k = its max similarity to any EARLIER question."""
    curve = [0.0]  # first question has no predecessor
    for i in range(1, len(order)):
        prev = order[:i]
        cur = order[i]
        curve.append(float(q_to_q[cur, prev].max()))
    return curve


def run_analysis(questions, chunks):
    q_to_c, q_to_q = build_similarity_matrix(questions, chunks)
    threshold = COVERAGE_THRESHOLD or auto_threshold(q_to_c)

    n = len(questions)
    rng = np.random.default_rng(RANDOM_SEED)

    cov_runs, red_runs = [], []
    for _ in range(N_SHUFFLES):
        order = list(rng.permutation(n))
        cov_runs.append(coverage_curve(order, q_to_c, threshold))
        red_runs.append(redundancy_curve(order, q_to_q))

    cov = np.array(cov_runs)   # shuffles x k
    red = np.array(red_runs)
    return {
        "k":            np.arange(1, n + 1),
        "cov_mean":     cov.mean(axis=0),
        "cov_std":      cov.std(axis=0),
        "red_mean":     red.mean(axis=0),
        "marginal":     np.diff(np.concatenate([[0], cov.mean(axis=0)])),
        "threshold":    threshold,
        "n_chunks":     len(chunks),
    }


# ── ELBOW DETECTION ────────────────────────────────────
def find_elbow(k: np.ndarray, y: np.ndarray) -> int:
    """Geometric (Kneedle-style) elbow: the point on the curve furthest from
    the straight line joining the first and last points. Falls back gracefully
    if the optional 'kneed' library is present (more robust)."""
    try:
        from kneed import KneeLocator
        kl = KneeLocator(k, y, curve="concave", direction="increasing")
        if kl.knee:
            return int(kl.knee)
    except Exception:
        pass
    # Manual fallback
    p1 = np.array([k[0], y[0]])
    p2 = np.array([k[-1], y[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    dists = []
    for i in range(len(k)):
        pt = np.array([k[i], y[i]]) - p1
        proj = pt.dot(line) * line
        dists.append(np.linalg.norm(pt - proj))
    return int(k[int(np.argmax(dists))])


# ── ELBOW SUMMARY ─────────────────────────────────────
def summarize_elbow(results: dict, elbow_k: int) -> str:
    """Return a plain-English explanation of why elbow_k was chosen."""
    k        = results["k"]
    cov_mean = results["cov_mean"]
    marginal = results["marginal"]
    red_mean = results["red_mean"]
    n_chunks = results["n_chunks"]
    n_total  = int(k[-1])

    elbow_idx  = elbow_k - 1
    elbow_cov  = cov_mean[elbow_idx] * 100
    final_cov  = cov_mean[-1] * 100
    missed_cov = final_cov - elbow_cov

    # Average marginal coverage gain before the elbow (questions 1 … elbow_k)
    pre_avg  = float(marginal[:elbow_k].mean())  * 100 if elbow_k > 0         else 0.0
    # Average marginal coverage gain after the elbow (questions elbow_k+1 … n)
    post_avg = float(marginal[elbow_k:].mean())  * 100 if elbow_k < n_total   else 0.0

    elbow_red = red_mean[elbow_idx] * 100

    # Describe the redundancy level in plain words
    if elbow_red < 20:
        red_label = "low"
    elif elbow_red < 50:
        red_label = "moderate"
    else:
        red_label = "high"

    lines = [
        "",
        "── SATURATION SUMMARY " + "─" * 47,
        f"Sweet spot: {elbow_k} question{'s' if elbow_k != 1 else ''}  "
        f"→  {elbow_cov:.0f}% of {n_chunks} document chunks covered",
        "",
        "Why this point was chosen:",
        f"  1. Coverage gain  — Questions 1–{elbow_k} each added an average of "
        f"{pre_avg:.1f}% new coverage per question.",
        f"     After question {elbow_k}, that drops sharply to ~{post_avg:.1f}% per "
        f"question — a {pre_avg - post_avg:.1f} percentage-point fall.",
        f"  2. Redundancy     — At question {elbow_k}, new questions are already "
        f"{elbow_red:.0f}% similar ({red_label}) to the existing set,",
        f"     meaning they are largely rephrasing topics already covered.",
        f"  3. Remaining gain — Using all {n_total} questions would add only "
        f"{missed_cov:.0f}% more coverage ({final_cov:.0f}% total),",
        f"     not worth the extra {n_total - elbow_k} questions.",
        "",
        "Detection method:",
        f"  The elbow was found geometrically: every point on the mean coverage",
        f"  curve was measured against the straight line from (1, {cov_mean[0]*100:.0f}%)",
        f"  to ({n_total}, {final_cov:.0f}%). Question {elbow_k} sits furthest from that",
        f"  baseline, marking the sharpest change in slope (point of diminishing returns).",
        "─" * 69,
    ]
    return "\n".join(lines)


# ── PLOTTING ───────────────────────────────────────────
def plot(results, elbow_k, out_path):
    k = results["k"]
    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    # Coverage curve with std band
    ax1.plot(k, results["cov_mean"], color="#2563eb", lw=2.5,
             label="Semantic coverage (mean)", zorder=3)
    ax1.fill_between(k,
                     results["cov_mean"] - results["cov_std"],
                     results["cov_mean"] + results["cov_std"],
                     color="#2563eb", alpha=0.15,
                     label="+/- 1 std (ordering variance)")
    ax1.set_xlabel("Number of questions in the test set", fontsize=12)
    ax1.set_ylabel("Fraction of policy chunks covered", color="#2563eb", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.25, zorder=0)

    # Redundancy curve on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(k, results["red_mean"], color="#dc2626", lw=2, ls="--",
             label="Redundancy of newest question", zorder=3)
    ax2.set_ylabel("Redundancy (similarity to existing questions)",
                   color="#dc2626", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_ylim(0, 1.05)

    # Elbow marker
    elbow_y = results["cov_mean"][elbow_k - 1]
    ax1.axvline(elbow_k, color="#16a34a", lw=2, alpha=0.7, zorder=2)
    ax1.scatter([elbow_k], [elbow_y], color="#16a34a", s=120, zorder=5)
    ax1.annotate(f"  Sweet spot ≈ {elbow_k} questions\n  ({elbow_y*100:.0f}% coverage)",
                 xy=(elbow_k, elbow_y), xytext=(elbow_k + 0.6, elbow_y - 0.18),
                 color="#16a34a", fontsize=11, fontweight="bold")

    fig.suptitle("Question-Set Saturation Analysis", fontsize=15, fontweight="bold")
    ax1.set_title(f"{results['n_chunks']} policy chunks · coverage threshold "
                  f"{results['threshold']:.3f} · {N_SHUFFLES} orderings averaged",
                  fontsize=10, color="#555")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Graph saved to {out_path}")


# ── IO ─────────────────────────────────────────────────
def load_questions(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return [q["question"] for q in data["questions"]]
    # CSV with a 'question' column
    with path.open() as f:
        return [r["question"] for r in csv.DictReader(f) if r.get("question")]


def main(doc_path, questions_path, out_path="saturation.png"):
    md = Path(doc_path).read_text()
    chunks = chunk_markdown_by_heading(md)
    questions = load_questions(Path(questions_path))

    print(f"Backend     : {EMBED_BACKEND}")
    print(f"Questions   : {len(questions)}")
    print(f"Doc chunks  : {len(chunks)}")

    results = run_analysis(questions, chunks)
    elbow_k = find_elbow(results["k"], results["cov_mean"])

    print(f"Threshold   : {results['threshold']:.3f}")
    print(f"Elbow (sweet spot): {elbow_k} questions "
          f"({results['cov_mean'][elbow_k-1]*100:.0f}% coverage)")
    print(f"Full set    : {len(questions)} questions "
          f"({results['cov_mean'][-1]*100:.0f}% coverage)")
    print(summarize_elbow(results, elbow_k))

    plot(results, elbow_k, out_path)
    return results, elbow_k


if __name__ == "__main__":
    main(
        "policies/HR_POLICY_LEAVE.markdown",          # your policy document
        "questions/HR_POLICY_LEAVE_doc.json",    # the JSON from qn_gen_personal.py
        "saturation.png"                        # output graph saved here
    )
