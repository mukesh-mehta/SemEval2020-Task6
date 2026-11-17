# DeftEval — definition extraction tools

[![Tests](https://github.com/mukesh-mehta/SemEval2020-Task6/actions/workflows/ci.yml/badge.svg)](https://github.com/mukesh-mehta/SemEval2020-Task6/actions/workflows/ci.yml)

Research code for SemEval 2020 Task 6, with a maintained, dependency-free Python
toolkit for reading DEFT token files and evaluating BIO spans. The original
sentence-classification and sequence-labeling models remain under `src/`.

## Quick start

Use Python 3.12. Conversion/evaluation requires only the standard library;
pytest is needed for development.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python deft_tools.py convert examples/sample.deft sample.jsonl
python deft_tools.py evaluate sample.jsonl sample.jsonl
```

Self-evaluation returns span precision/recall/F1 and token accuracy of 1.0. This
checks file handling, not a trained model. The converter refuses to overwrite an
existing output file. On Windows activate with `.venv\Scripts\activate`.

## Formats and evaluation semantics

DEFT rows are tab-separated: `token`, `source`, `start`, `end`, `BIO tag`, followed
by optional annotation columns. Blank lines separate sequences. The parser handles
CRLF, trailing blank lines, UTF-8 BOM, and a final sequence without a newline.
Offsets are preserved as supplied, not converted between inclusive/exclusive
character conventions. Invalid rows fail with their source line number.

For four-column unlabeled input use `convert ... --unlabeled`; tags are filled
with `O`. Output is JSONL, with one object per sequence:

```json
{"tokens":["triangle"],"sources":["example.txt"],"starts":[0],"ends":[7],"tags":["B-Term"]}
```

To score predictions, produce JSONL records with the **same tokens and sequence
order** and predicted `tags`. The evaluator requires exact token alignment and
matching lengths. It accepts any nonempty BIO entity type, rejects orphan or
mismatched `I-` tags, and matches complete `(type, start, end)` token spans.
Metrics are micro-averaged across sequences. If there are no positive spans,
precision/recall/F1 are 0; token accuracy is still reported.

This is a strict general BIO evaluator, **not the official DeftEval scorer**.
It does not apply Task 2's official label subset, relation rules, or preprocessing.
Use the official resources when reporting competition-comparable results.

## Features and remaining scope

| Capability | Status |
| --- | --- |
| DEFT token-file parsing | Implemented |
| Labeled and unlabeled inputs | Implemented |
| UTF-8 JSONL export | Implemented |
| Offset and row validation | Implemented |
| Token-alignment validation | Implemented |
| Strict BIO span extraction | Implemented |
| Micro span precision/recall/F1 | Implemented |
| Token accuracy and regression tests | Implemented |
| Current Transformers training pipeline | Planned |
| Official-score integration | Planned |
| Subtask 3 relation classification | Planned |
| Model cards and reproducible benchmark runs | Planned |

## Dataset and original research

Obtain the corpus from [Adobe Research's DEFT repository](https://github.com/adobe-research/deft_corpus)
and follow its license and task instructions. The included example is synthetic;
no corpus download happens automatically. `.gitmodules` records the historical
corpus location, but the quick start does not rely on a populated submodule.

The original project tackled sentence classification (subtask 1) and sequence
labeling (subtask 2), with reported ranks 41 and 46. Those historical claims are
preserved in [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md), not rerun results.
`src/` and `environment.yml` describe old Python 3.7, torchtext, Transformers,
spaCy, and CUDA experiments. They are not the maintained installation and are not
covered by the current CI. The new converter/evaluator imports none of them.

Tests cover exact span boundaries, misalignment, malformed tags/offsets, and
line-ending edge cases. New maintained code uses [MIT](LICENSE). Existing research
code includes external adaptations and is not relicensed by this refresh; see
[NOTICE.md](NOTICE.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
