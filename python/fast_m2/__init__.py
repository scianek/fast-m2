"""m2scorer public API.

    from m2scorer import score, score_files

compute_metrics(hypotheses, m2_path)  ->  {"precision": ..., "recall": ..., "f_0.5": ...}
score_files(hyp_path, m2_path)        ->  same dict
"""

from __future__ import annotations

from pathlib import Path

from ._core import batch_multi_pre_rec_f1
from .io import load_hypotheses, load_m2


def score(
    hypotheses: list[str],
    m2_path: str | Path,
    *,
    beta: float = 0.5,
    max_unchanged_words: int = 2,
    ignore_whitespace_casing: bool = False,
) -> dict[str, float]:
    """Score a list of hypothesis sentences against an M2 gold file.

    Parameters
    ----------
    hypotheses : list[str]
        System output sentences, one per line (already tokenised the same way
        as the M2 source sentences).
    m2_path : str | Path
        Path to the M2-format gold file (.m2 or .m2.gz).
    beta : float
        Beta for F_beta.  Default 0.5 (standard for GEC).
    max_unchanged_words : int
        Maximum unchanged words when grouping edits into spans.  Default 2.
    ignore_whitespace_casing : bool
        Ignore edits that only differ in whitespace / casing.

    Returns
    -------
    dict with keys "precision", "recall", f"f_{beta}"
    """
    sources, gold_edits = load_m2(m2_path)

    if len(hypotheses) != len(sources):
        raise ValueError(
            f"Number of hypotheses ({len(hypotheses)}) does not match "
            f"number of source sentences in M2 file ({len(sources)})."
        )

    p, r, f = batch_multi_pre_rec_f1(
        hypotheses,
        sources,
        gold_edits,
        max_unchanged_words,
        beta,
        ignore_whitespace_casing,
    )

    return {"precision": p, "recall": r, f"f_{beta}": f}


def score_files(
    hypothesis_path: str | Path,
    m2_path: str | Path,
    *,
    beta: float = 0.5,
    max_unchanged_words: int = 2,
    ignore_whitespace_casing: bool = False,
) -> dict[str, float]:
    """Convenience wrapper: load hypotheses from a file, then call compute_metrics."""
    hypotheses = load_hypotheses(hypothesis_path)
    return score(
        hypotheses,
        m2_path,
        beta=beta,
        max_unchanged_words=max_unchanged_words,
        ignore_whitespace_casing=ignore_whitespace_casing,
    )
