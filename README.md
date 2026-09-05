# fast-m2

A fast, modern Rust-backed implementation of the **MaxMatch ($M^2$) scorer** for Grammatical Error Correction (GEC) evaluation.

It is a drop-in replacement for the official Python 2 [nusnlp/m2scorer](https://github.com/nusnlp/m2scorer) (Dahlmeier & Ng, 2012), providing **100% bit-for-bit numerical parity** on standard benchmarks (CoNLL-2014, BEA-19) while running significantly faster.

---

## Installation

### Option 1: Install Pre-built Wheels (No Rust Compiler Needed)

Pre-compiled binary wheels for Linux, macOS and Windows are attached to the latest release:

```bash
pip install fast-m2 --find-links https://github.com/scianek/fast-m2/releases/expanded_assets/v0.1.0
```

Using `uv`:
```bash
uv pip install fast-m2 --find-links https://github.com/scianek/fast-m2/releases/expanded_assets/v0.1.0
```

### Option 2: Install from Source via Git (Requires Rust Toolchain)

If you have Rust (`cargo`) installed on your system:

```bash
pip install git+https://github.com/scianek/fast-m2.git
```

### Option 3: Local Editable Build (For Development)

```bash
git clone https://github.com/scianek/fast-m2.git
cd fast-m2
maturin develop --release
```

---

## Quickstart

### 1. CLI Usage

Installing `fast-m2` provides a command line interface:

```bash
fast-m2 hypotheses.txt official-2014.combined.m2
```

Output:
```text
Precision   : 0.7521
Recall      : 0.4056
F_0.5       : 0.6423
```

#### CLI Options:

* `--beta, -b`: Beta value for $F_\beta$ (default: `0.5`).
* `--max-unchanged-words, -m`: Max unchanged tokens allowed when merging edits (default: `2`).
* `--ignore-whitespace-casing`: Ignore corrections that only alter casing/whitespace.
* `--json`: Emit machine-readable JSON metrics:

---

### 2. Python API

Use the unified `score()` function. It accepts file paths (`str` or `pathlib.Path`), lists of sentences or single strings.

#### From Files

```python
import fast_m2

result = fast_m2.score("hypotheses.txt", "official-2014.combined.m2")

print(result.precision)  # 0.7521...
print(result.recall)     # 0.4056...
print(result.f_beta)     # 0.6423...
```

#### In-Memory Sentences

```python
import fast_m2

hypotheses = [
    "This is a corrected sentence .",
    "Another system output sentence .",
]

result = fast_m2.score(hypotheses, "gold.m2", beta=0.5)

# Supports dict-like access:
print(result["f_0.5"])

# Supports tuple unpacking:
p, r, f = result
```

---

## Citation & Reference

The underlying MaxMatch algorithm was proposed by Dahlmeier & Ng:

> Daniel Dahlmeier and Hwee Tou Ng. 2012. **Better Evaluation for Grammatical Error Correction**. In *Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2012)*, pages 568–572.

The canonical Python 2 reference implementation is maintained at [nusnlp/m2scorer](https://github.com/nusnlp/m2scorer).

---

## License

GNU General Public License v3.0 (GPL-3.0), matching the original NUS $M^2$ Scorer.
