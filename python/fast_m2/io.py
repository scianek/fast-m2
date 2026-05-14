from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path


def _smart_open(path: str | Path, mode: str = "r"):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode, 1)
    return open(p, mode, encoding="utf-8")


def _paragraphs(lines: list[str]) -> Iterator[list[str]]:
    """Yield non-empty groups of lines separated by blank lines."""
    block: list[str] = []
    for line in lines:
        if line == "\n":
            if block:
                yield block
                block = []
        else:
            block.append(line)
    if block:
        yield block


def load_m2(path: str | Path) -> tuple[list[str], list[dict]]:
    """Parse an M2-format file.

    Returns
    -------
    source_sentences : list[str]
    gold_edits       : list[dict[int, list[tuple[int, int, str, list[str]]]]]
        One entry per sentence.  Each dict maps annotator_id to a list of
        (start_offset, end_offset, original_text, [correction, ...]) tuples.
        Noop edits are filtered out (they carry no information for scoring).
    """
    with _smart_open(path, "r") as fh:
        raw = fh.read()

    source_sentences: list[str] = []
    gold_edits: list[dict] = []

    for block in _paragraphs(raw.splitlines(keepends=True)):
        block = [ln.rstrip("\n") for ln in block]

        # A block may contain multiple sentences separated by 'S ' lines
        sentences = [ln[2:].strip() for ln in block if ln.startswith("S ")]
        assert sentences, f"Block has no S line: {block!r}"

        # Collect annotations keyed by annotator id
        annotations: dict[int, list] = {}
        for ln in block:
            if ln.startswith("I ") or ln.startswith("S "):
                continue
            if not ln.startswith("A "):
                continue

            fields = ln[2:].split("|||")
            offsets = fields[0].split()
            start = int(offsets[0])
            end = int(offsets[1])
            etype = fields[1]

            if etype == "noop":
                # noop means "no error in this sentence for this annotator";
                # we still need the annotator to exist so that multi-annotator
                # scoring doesn't skip them entirely.
                annotator = int(fields[5])
                if annotator not in annotations:
                    annotations[annotator] = []
                continue

            corrections = [
                c.strip() if c.strip() != "-NONE-" else ""
                for c in fields[2].split("||")
            ]
            annotator = int(fields[5])
            if annotator not in annotations:
                annotations[annotator] = []

            # Reconstruct the original token span from the joined sentence tokens
            all_tokens = " ".join(sentences).split()
            original = " ".join(all_tokens[start:end])

            annotations[annotator].append((start, end, original, corrections))

        # Split multi-sentence blocks per token offset (mirrors Python scorer)
        tok_offset = 0
        for sentence in sentences:
            tok_offset += len(sentence.split())
            source_sentences.append(sentence)
            this_edits: dict[int, list] = {}
            for ann_id, ann_list in annotations.items():
                this_edits[ann_id] = [
                    e
                    for e in ann_list
                    if 0 <= e[0] <= tok_offset and 0 <= e[1] <= tok_offset
                ]
            if not this_edits:
                this_edits[0] = []
            gold_edits.append(this_edits)

    return source_sentences, gold_edits


def load_hypotheses(path: str | Path) -> list[str]:
    """Load system output: one hypothesis sentence per line."""
    with _smart_open(path, "r") as fh:
        return [ln.strip() for ln in fh.readlines()]
