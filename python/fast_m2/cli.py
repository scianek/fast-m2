from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fast-m2",
        description="Fast M2 Scorer for Grammatical Error Correction evaluation.",
    )

    parser.add_argument(
        "hypothesis",
        type=Path,
        help="Path to system output file (one sentence per line).",
    )
    parser.add_argument(
        "gold",
        type=Path,
        help="Path to gold M2 file (.m2 or .m2.gz).",
    )
    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=0.5,
        help="Beta value for F_beta (default: 0.5).",
    )
    parser.add_argument(
        "--max-unchanged-words",
        "--max_unchanged_words",
        "-m",
        dest="max_unchanged_words",
        type=int,
        default=2,
        help="Maximum unchanged words when grouping edits into spans (default: 2).",
    )
    parser.add_argument(
        "--ignore-whitespace-casing",
        "--ignore_whitespace_casing",
        action="store_true",
        help="Ignore edits that only affect whitespace and casing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw metrics as JSON.",
    )

    args = parser.parse_args(argv)

    if not args.hypothesis.is_file():
        print(
            f"Error: hypothesis file '{args.hypothesis}' does not exist.",
            file=sys.stderr,
        )
        return 1
    if not args.gold.is_file():
        print(f"Error: gold M2 file '{args.gold}' does not exist.", file=sys.stderr)
        return 1

    try:
        results = score(
            args.hypothesis,
            args.gold,
            beta=args.beta,
            max_unchanged_words=args.max_unchanged_words,
            ignore_whitespace_casing=args.ignore_whitespace_casing,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results.as_dict(), indent=2))
    else:
        print(f"Precision   : {results.precision:.4f}")
        print(f"Recall      : {results.recall:.4f}")
        print(f"F_{args.beta:<9.1f}: {results.f_beta:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
