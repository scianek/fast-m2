"""Fast M2 Scorer for Grammatical Error Correction evaluation."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ._core import batch_multi_pre_rec_f1
from .io import load_hypotheses, load_m2

__all__ = ["load_hypotheses", "load_m2", "score"]


def _resolve_hypotheses(
    hypotheses: str | Path | Sequence[str] | Iterable[str],
) -> list[str]:
    # 1. Path object -> definitely a file
    if isinstance(hypotheses, Path):
        return load_hypotheses(hypotheses)

    # 2. String input
    if isinstance(hypotheses, str):
        # Multi-line string -> in-memory sentences
        if "\n" in hypotheses:
            return [line.strip() for line in hypotheses.splitlines() if line.strip()]

        # Existing file path
        if os.path.isfile(hypotheses):
            return load_hypotheses(hypotheses)

        # Fallback: single sentence string
        return [hypotheses.strip()]

    # 3. Iterable/Sequence of sentences (list, tuple, etc.)
    return [str(h).strip() for h in hypotheses]


def _resolve_m2(m2: str | Path) -> tuple[list[str], list[dict[int, list[Any]]]]:
    if isinstance(m2, Path) or (isinstance(m2, str) and os.path.isfile(m2)):
        return load_m2(m2)

    raise FileNotFoundError(f"Gold M2 file not found: {m2!r}")


def score(
    hypotheses: str | Path | Sequence[str] | Iterable[str],
    m2: str | Path,
    *,
    beta: float = 0.5,
    max_unchanged_words: int = 2,
    ignore_whitespace_casing: bool = False,
) -> dict[str, float]:
    """Score system hypotheses against an M2 reference.

    Parameters
    ----------
    hypotheses : str | Path | Sequence[str] | Iterable[str]
        System output. Can be:
          - A Path object to a hypothesis file
          - A string path to an existing file
          - A list / sequence of tokenized sentences
          - A single sentence string
    m2 : str | Path
        Path to gold M2 file (.m2 or .m2.gz).
    beta : float, default=0.5
        Beta value for F_beta (0.5 weights precision twice as much as recall).
    max_unchanged_words : int, default=2
        Maximum unchanged words allowed when grouping edits into spans.
    ignore_whitespace_casing : bool, default=False
        Ignore edits that only affect whitespace and casing.

    Returns
    -------
    dict[str, float]
        Dictionary with keys "precision", "recall", and f"f_{beta}".
    """
    hyp_list = _resolve_hypotheses(hypotheses)
    sources, gold_edits = _resolve_m2(m2)

    if len(hyp_list) != len(sources):
        raise ValueError(
            f"Sentence count mismatch: got {len(hyp_list)} hypotheses, "
            f"but found {len(sources)} source sentences in {m2}."
        )

    p, r, f = batch_multi_pre_rec_f1(
        hyp_list,
        sources,
        gold_edits,
        max_unchanged_words,
        beta,
        ignore_whitespace_casing,
    )

    return {"precision": p, "recall": r, f"f_{beta}": f}
