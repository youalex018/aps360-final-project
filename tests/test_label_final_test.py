"""Tests for the final-test CSV labeler."""
from __future__ import annotations

from pathlib import Path

from label_final_test import (
    clear_label,
    load_rows,
    parse_command,
    progress_counts,
    run_labeler,
    save_rows,
    set_label,
    unlabeled_indices,
)


def _write_csv(path: Path) -> None:
    path.write_text(
        "match_id,message_order,text,label,notes\n"
        "M-1,1,gg wp,,\n"
        "M-1,2,you suck,0,\n"
        "M-1,3,uninstall,,\n",
        encoding="utf-8",
    )


def test_load_save_roundtrip(tmp_path: Path):
    path = tmp_path / "final_chat.csv"
    _write_csv(path)
    rows = load_rows(path)
    assert len(rows) == 3
    assert unlabeled_indices(rows) == [0, 2]
    set_label(rows, 0, "0", notes="banter")
    save_rows(path, rows)
    reloaded = load_rows(path)
    assert reloaded[0]["label"] == "0"
    assert reloaded[0]["notes"] == "banter"


def test_parse_command_variants():
    assert parse_command("0") == ("0", "")
    assert parse_command("1") == ("1", "")
    assert parse_command("s")[0] == "s"
    assert parse_command("n unsure") == ("n", "unsure")
    assert parse_command("note edge case") == ("n", "edge case")
    assert parse_command("nope")[0] == "?"


def test_progress_and_clear(tmp_path: Path):
    path = tmp_path / "final_chat.csv"
    _write_csv(path)
    rows = load_rows(path)
    set_label(rows, 2, "1")
    labeled, total, toxic, safe = progress_counts(rows)
    assert (labeled, total, toxic, safe) == (2, 3, 1, 1)
    clear_label(rows, 2)
    assert 2 in unlabeled_indices(rows)


def test_run_labeler_applies_numbers(tmp_path: Path):
    path = tmp_path / "final_chat.csv"
    _write_csv(path)
    answers = iter(["0", "1", "q"])

    def fake_input(_prompt: str = "") -> str:
        return next(answers)

    code = run_labeler(path, input_fn=fake_input)
    assert code == 0
    rows = load_rows(path)
    assert rows[0]["label"] == "0"
    assert rows[2]["label"] == "1"
    assert rows[1]["label"] == "0"
