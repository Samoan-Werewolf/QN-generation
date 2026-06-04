"""
qn_gen_from_chunks.py
─────────────────────
Generate HR Q&A from a pre-chunked JSON file instead of a raw markdown file.

Expects the input JSON to be one of two shapes:

  Shape A – plain list:
    [
      {"title": "Annual Leave",  "content": "## Annual Leave\n\n..."},
      {"title": "Sick Leave",    "content": "## Sick Leave\n\n..."},
      ...
    ]

  Shape B – wrapped object:
    {
      "source": "HR_POLICY_LEAVE.md",   ← optional, used in output filename
      "chunks": [ ... ]                  ← same list as Shape A
    }

Usage:
    python qn_gen_from_chunks.py <chunks.json> [options]

Options:
    -n, --questions N     Questions per chunk (default: 5)
    -o, --output PATH     Output file path   (default: questions/<source>_from_chunks.json)
    --dry-run             Skip API calls; fill with placeholder questions
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Import shared utilities from the main app ──────────────────────
# qn_gen_chunk.py defines a Flask app at module level, but app.run()
# is guarded by __name__ == "__main__", so importing is safe.
from qn_gen_chunk import (
    OUTPUT_DIR,
    QUESTIONS_PER_CHUNK,
    generate_qa_for_chunk,
)
# ───────────────────────────────────────────────────────────────────


def load_chunks(json_path: Path) -> tuple[list[dict], str]:
    """
    Load chunks from a JSON file.
    Returns (chunks, source_name) where source_name is used in the output.
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        chunks = raw
        source = json_path.stem
    elif isinstance(raw, dict):
        chunks = raw.get("chunks", [])
        source = raw.get("source", json_path.stem)
    else:
        raise ValueError(
            f"Unexpected JSON structure in '{json_path}'. "
            "Expected a list of chunks or a dict with a 'chunks' key."
        )

    # Validate each chunk has the required keys
    for i, chunk in enumerate(chunks):
        if "title" not in chunk or "content" not in chunk:
            raise ValueError(
                f"Chunk {i} is missing 'title' or 'content' key: {chunk}"
            )

    return chunks, source


def process_chunks(
    chunks: list[dict],
    source: str,
    n: int,
    dry_run: bool,
) -> dict:
    """Run Q&A generation over a list of pre-built chunks."""
    sections = []
    for chunk in chunks:
        qa_list = generate_qa_for_chunk(chunk, n=n, dry_run=dry_run)
        sections.append(
            {
                "section": chunk["title"],
                "question_count": len(qa_list),
                "qa_pairs": qa_list,
            }
        )
        time.sleep(0.3)

    return {
        "policy_source": source,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_sections": len(sections),
        "total_questions": sum(s["question_count"] for s in sections),
        "sections": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HR Q&A from a pre-chunked JSON file."
    )
    parser.add_argument("chunks_file", help="Path to the input chunks JSON file.")
    parser.add_argument(
        "-n", "--questions",
        type=int,
        default=QUESTIONS_PER_CHUNK,
        dest="n",
        help=f"Questions to generate per chunk (default: {QUESTIONS_PER_CHUNK}).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        dest="output",
        help="Output file path. Defaults to questions/<source>_from_chunks.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Azure API calls and fill with placeholder questions.",
    )
    args = parser.parse_args()

    chunks_path = Path(args.chunks_file)
    if not chunks_path.exists():
        print(f"Error: file not found: '{chunks_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading chunks from '{chunks_path}' …")
    chunks, source = load_chunks(chunks_path)
    print(f"  Found {len(chunks)} chunk(s)  |  source: '{source}'")

    if not chunks:
        print("No chunks to process. Exiting.")
        sys.exit(0)

    print(
        f"Generating {args.n} question(s) per chunk"
        + (" [DRY RUN]" if args.dry_run else "")
        + " …"
    )
    output = process_chunks(chunks, source=source, n=args.n, dry_run=args.dry_run)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem if "." in source else source
        out_path = OUTPUT_DIR / f"{stem}_from_chunks.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Sections  : {output['total_sections']}")
    print(f"  Questions : {output['total_questions']}")
    print(f"  Output    : {out_path}")


if __name__ == "__main__":
    main()
