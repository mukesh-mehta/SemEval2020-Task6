import pytest
from deft_tools import evaluate, parse_deft, spans


def test_parse_crlf_and_trailing_blank_lines(tmp_path):
    path = tmp_path / "sample.deft"
    path.write_bytes(b"A\tsource\t0\t0\tB-Term\r\nterm\tsource\t2\t5\tI-Term\r\n\r\n\r\n")
    records = parse_deft(path)
    assert len(records) == 1
    assert records[0]["tokens"] == ["A", "term"]
    assert records[0]["starts"] == [0, 2]
    assert evaluate(records, records)["span_f1"] == 1


@pytest.mark.parametrize("row", ["bad", "x\tf\ta\t2\tO", "x\tf\t2\t1\tO", "x\tf\t0\t1\tX"])
def test_malformed_rows_fail_with_line(tmp_path, row):
    path = tmp_path / "bad.deft"
    path.write_text(row)
    with pytest.raises(ValueError, match="Line 1"):
        parse_deft(path)


def test_unlabeled_and_final_sequence(tmp_path):
    path = tmp_path / "test.deft"
    path.write_text("word\tf\t0\t3")
    assert parse_deft(path, unlabeled=True)[0]["tags"] == ["O"]


def test_boundaries_and_metrics():
    assert spans(["B-Term", "I-Term", "B-Term", "O"]) == {("Term", 0, 2), ("Term", 2, 3)}
    gold = [{"tokens": ["a", "b", "c"], "tags": ["B-Term", "I-Term", "O"]}]
    pred = [{"tokens": ["a", "b", "c"], "tags": ["B-Term", "O", "O"]}]
    result = evaluate(gold, pred)
    assert result["span_f1"] == 0
    assert result["token_accuracy"] == pytest.approx(2 / 3)
    pred[0]["tokens"] = ["other"]
    with pytest.raises(ValueError, match="alignment"):
        evaluate(gold, pred)


@pytest.mark.parametrize("tags", [["I-Term"], ["B-Term", "I-Definition"], ["B-"], [None]])
def test_invalid_bio(tags):
    with pytest.raises(ValueError):
        spans(tags)


@pytest.mark.parametrize("record", [[], {"tokens": ["a"], "tags": "O"},
                                  {"tokens": [1], "tags": ["O"]}])
def test_invalid_json_schema(record):
    with pytest.raises(ValueError):
        evaluate([record], [record])
