"""Dependency-free DEFT conversion and strict BIO span evaluation."""
import argparse
import json
from pathlib import Path


def parse_deft(path, *, unlabeled=False):
    records, rows = [], []

    def flush():
        if rows:
            records.append({key: [row[i] for row in rows] for i, key in
                            enumerate(["tokens", "sources", "starts", "ends", "tags"])})
            rows.clear()

    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                flush()
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < (4 if unlabeled else 5):
                raise ValueError(f"Line {number}: expected tab-separated token, source, start, end, tag")
            try:
                start, end = int(parts[2]), int(parts[3])
            except ValueError as exc:
                raise ValueError(f"Line {number}: character offsets must be integers") from exc
            if not parts[0] or not parts[1] or start < 0 or end < start:
                raise ValueError(f"Line {number}: empty token/source or invalid offsets")
            tag = "O" if unlabeled else parts[4]
            if tag != "O" and (len(tag) < 3 or tag[:2] not in ("B-", "I-")):
                raise ValueError(f"Line {number}: invalid BIO tag {tag!r}")
            rows.append((parts[0], parts[1], start, end, tag))
    flush()
    if not records:
        raise ValueError("No token sequences found")
    return records


def spans(tags):
    """Return (type, start, exclusive end), rejecting orphan/mismatched I tags."""
    result, active, start = set(), None, 0
    for i, tag in enumerate([*tags, "O"]):
        if not isinstance(tag, str) or (tag != "O" and (tag[:2] not in ("B-", "I-") or len(tag) < 3)):
            raise ValueError(f"Invalid BIO tag at token {i}: {tag!r}")
        if tag.startswith("I-"):
            if active != tag[2:]:
                raise ValueError(f"Orphan or mismatched I tag at token {i}")
            continue
        if active is not None:
            result.add((active, start, i))
            active = None
        if tag.startswith("B-"):
            active, start = tag[2:], i
    return result


def evaluate(gold, predicted):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Gold and predictions must have the same nonzero sequence count")
    tp = fp = fn = correct = total = 0
    for i, (expected, actual) in enumerate(zip(gold, predicted)):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise ValueError(f"Sequence {i}: expected JSON objects")
        tokens = expected.get("tokens")
        if (not isinstance(tokens, list) or not tokens or
                any(not isinstance(token, str) or not token for token in tokens) or
                tokens != actual.get("tokens")):
            raise ValueError(f"Sequence {i}: token alignment differs or is empty")
        if any(not isinstance(row.get("tags"), list) or len(row["tags"]) != len(tokens)
               for row in (expected, actual)):
            raise ValueError(f"Sequence {i}: token/tag lengths differ")
        truth, guess = spans(expected["tags"]), spans(actual["tags"])
        tp += len(truth & guess)
        fp += len(guess - truth)
        fn += len(truth - guess)
        correct += sum(a == b for a, b in zip(expected["tags"], actual["tags"]))
        total += len(tokens)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"span_precision": precision, "span_recall": recall,
            "span_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "token_accuracy": correct / total, "true_positive": tp,
            "false_positive": fp, "false_negative": fn, "tokens": total}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--unlabeled", action="store_true")
    score = commands.add_parser("evaluate")
    score.add_argument("gold", type=Path)
    score.add_argument("predictions", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "convert":
            records = parse_deft(args.input, unlabeled=args.unlabeled)
            with args.output.open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"sequences": len(records), "output": str(args.output)}))
        else:
            print(json.dumps(evaluate(read_jsonl(args.gold), read_jsonl(args.predictions)), indent=2))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
