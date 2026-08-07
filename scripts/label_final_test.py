"""Interactive terminal labeler for ``data/final_test/final_chat.csv``.

Shows each unlabeled message with nearby same-match context. Type ``0``
(non-toxic) or ``1`` (toxic) to write the label immediately; optional notes
and skip/back/quit commands keep long sessions manageable.

0 non-toxic
1 toxic
s skip
b undo previous label
n … add/edit notes
c clear this row's label
q quit
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

from toxic_chat import config

DEFAULT_CSV = config.ROOT / "data" / "final_test" / "final_chat.csv"
FIELDNAMES = ["match_id", "message_order", "text", "label", "notes"]
CONTEXT_RADIUS = 2


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        missing = [name for name in FIELDNAMES if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(missing)}")
        rows = []
        for row in reader:
            rows.append({name: (row.get(name) or "") for name in FIELDNAMES})
    return rows


def save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically rewrite the CSV so a crash mid-write cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
        temp_name = handle.name
    Path(temp_name).replace(path)


def unlabeled_indices(rows: list[dict[str, str]]) -> list[int]:
    return [index for index, row in enumerate(rows) if not str(row.get("label", "")).strip()]


def progress_counts(rows: list[dict[str, str]]) -> tuple[int, int, int, int]:
    total = len(rows)
    labeled = total - len(unlabeled_indices(rows))
    toxic = sum(1 for row in rows if str(row.get("label", "")).strip() == "1")
    safe = sum(1 for row in rows if str(row.get("label", "")).strip() == "0")
    return labeled, total, toxic, safe


def set_label(
    rows: list[dict[str, str]],
    index: int,
    label: str,
    *,
    notes: str | None = None,
) -> None:
    if label not in {"0", "1"}:
        raise ValueError(f"label must be '0' or '1', got {label!r}")
    rows[index]["label"] = label
    if notes is not None:
        rows[index]["notes"] = notes


def clear_label(rows: list[dict[str, str]], index: int) -> None:
    rows[index]["label"] = ""


def format_context(rows: list[dict[str, str]], index: int, radius: int = CONTEXT_RADIUS) -> str:
    current = rows[index]
    match_id = current["match_id"]
    lines = []
    for offset in range(-radius, radius + 1):
        neighbor = index + offset
        if neighbor < 0 or neighbor >= len(rows):
            continue
        row = rows[neighbor]
        if row["match_id"] != match_id:
            continue
        marker = ">" if neighbor == index else " "
        label = str(row.get("label", "")).strip() or "-"
        order = row["message_order"]
        text = row["text"].replace("\n", " ")
        lines.append(f"{marker} [{order}|{label}] {text}")
    return "\n".join(lines)


def print_prompt(rows: list[dict[str, str]], index: int, queue_pos: int, queue_len: int) -> None:
    labeled, total, toxic, safe = progress_counts(rows)
    row = rows[index]
    print()
    print("=" * 72)
    print(
        f"Unlabeled {queue_pos}/{queue_len}  |  overall {labeled}/{total} "
        f"(0={safe}, 1={toxic})"
    )
    print(f"match={row['match_id']}  order={row['message_order']}")
    print("-" * 72)
    print(format_context(rows, index))
    print("-" * 72)
    notes = str(row.get("notes", "")).strip()
    if notes:
        print(f"notes: {notes}")
    print("0=non-toxic  1=toxic  s=skip  b=back  n=note  c=clear  q=quit")


def parse_command(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if not text:
        return "", ""
    lower = text.lower()
    aliases = {
        "0": "0",
        "1": "1",
        "s": "s",
        "skip": "s",
        "b": "b",
        "back": "b",
        "c": "c",
        "clear": "c",
        "q": "q",
        "quit": "q",
        "exit": "q",
    }
    if lower in aliases:
        return aliases[lower], ""
    if lower.startswith("n ") or lower == "n" or lower.startswith("note"):
        if lower == "n" or lower == "note":
            return "n", ""
        if lower.startswith("note "):
            return "n", text[5:].strip()
        return "n", text[2:].strip()
    return "?", text


def run_labeler(
    path: Path,
    *,
    relabel: bool = False,
    start: int = 1,
    input_fn=input,
) -> int:
    rows = load_rows(path)
    if not rows:
        print("CSV is empty.")
        return 0

    if relabel:
        queue = list(range(len(rows)))
    else:
        queue = unlabeled_indices(rows)

    if start < 1:
        raise ValueError("--start must be >= 1")
    # Convert 1-based message display start into queue offset when possible.
    if start > 1 and not relabel:
        # start refers to absolute row number (1-based) in the CSV body.
        absolute = start - 1
        if absolute in queue:
            queue = queue[queue.index(absolute) :]
        else:
            queue = [index for index in queue if index >= absolute]

    if not queue:
        labeled, total, toxic, safe = progress_counts(rows)
        print(f"Nothing left to label. {labeled}/{total} done (0={safe}, 1={toxic}).")
        return 0

    cursor = 0
    history: list[int] = []

    print(f"Labeling {path}")
    print("Rubric: 1=targeted insult/harassment/abuse; 0=tactics/banter/ambiguous.")
    print("Labels are saved to disk after every change.")

    while 0 <= cursor < len(queue):
        index = queue[cursor]
        print_prompt(rows, index, cursor + 1, len(queue))
        try:
            raw = input_fn("> ")
        except EOFError:
            print()
            break

        command, payload = parse_command(raw)
        if command in {"0", "1"}:
            note = str(rows[index].get("notes", ""))
            set_label(rows, index, command, notes=note)
            save_rows(path, rows)
            history.append(index)
            cursor += 1
            continue
        if command == "s":
            cursor += 1
            continue
        if command == "b":
            if history:
                prev = history.pop()
                clear_label(rows, prev)
                save_rows(path, rows)
                if prev in queue:
                    cursor = queue.index(prev)
                else:
                    queue.insert(cursor, prev)
            elif cursor > 0:
                cursor -= 1
            else:
                print("Nothing to go back to.")
            continue
        if command == "c":
            clear_label(rows, index)
            rows[index]["notes"] = ""
            save_rows(path, rows)
            print("Cleared label/notes for this row.")
            continue
        if command == "n":
            note = payload
            if not note:
                try:
                    note = input_fn("note> ").strip()
                except EOFError:
                    print()
                    break
            rows[index]["notes"] = note
            save_rows(path, rows)
            print("Saved note. Now enter 0 or 1 (or another command).")
            continue
        if command == "q":
            break
        print("Unknown input. Type 0, 1, s, b, n, c, or q.")

    labeled, total, toxic, safe = progress_counts(rows)
    remaining = len(unlabeled_indices(rows))
    print()
    print(f"Stopped. {labeled}/{total} labeled (0={safe}, 1={toxic}); {remaining} blank.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV to label (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Walk every row, including ones that already have labels",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based CSV row number to begin from (header excluded)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_labeler(args.csv, relabel=args.relabel, start=args.start)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
