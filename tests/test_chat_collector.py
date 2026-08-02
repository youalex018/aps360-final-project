import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from collect_chat import (
    ChatMessage,
    OCRLine,
    anonymize_text,
    export_reviewed,
    extract_chat_messages,
    league_window_rank,
    new_visible_messages,
    preprocess_for_ocr,
    visible_overlap,
)


def line(text: str, top: int, confidence: float = 90.0) -> OCRLine:
    return OCRLine(
        text=text,
        confidence=confidence,
        left=10,
        top=top,
        width=300,
        height=16,
    )


def message(text: str, top: int) -> ChatMessage:
    return ChatMessage(
        text=text,
        channel="unknown",
        confidence=90.0,
        kind="player",
        left=10,
        top=top,
        width=200,
        height=16,
    )


class ParsingTests(unittest.TestCase):
    def test_parses_prefix_channel_and_wrapped_line(self):
        messages = extract_chat_messages(
            [
                line("[12:34] [All] Player One (Lux): this is a", 10),
                line("wrapped message", 28),
                line("Player Two (Garen): gg", 46),
            ]
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].channel, "all")
        self.assertEqual(messages[0].text, "this is a wrapped message")
        self.assertEqual(messages[0].kind, "player")
        self.assertEqual(messages[1].text, "gg")

    def test_keeps_unparsed_text_for_review(self):
        messages = extract_chat_messages([line("possibly OCR damaged text", 10)])

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].kind, "uncertain")

    def test_flags_system_notification(self):
        messages = extract_chat_messages([line("Blue turret destroyed", 10)])

        self.assertEqual(messages[0].kind, "system")


class ImagePipelineTests(unittest.TestCase):
    def test_synthetic_screenshot_preprocessing(self):
        image = np.zeros((80, 320, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        # Mid-gray "terrain" should be ignored; only bright glyphs kept.
        image[:, :] = (90, 95, 85, 255)
        cv2.putText(
            image,
            "Player: group dragon",
            (5, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "[Team]",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 180, 40, 255),
            1,
            cv2.LINE_AA,
        )

        processed = preprocess_for_ocr(image, cv2)

        self.assertEqual(processed.ndim, 2)
        self.assertGreater(processed.shape[0], image.shape[0])
        self.assertTrue(set(np.unique(processed)).issubset({0, 255}))
        # Light glyphs on dark UI should become dark text on a light page.
        self.assertGreater(float(processed.mean()), 200.0)


class ReconciliationTests(unittest.TestCase):
    def test_overlapping_visible_history_adds_only_suffix(self):
        previous = [message("ward dragon", 10), message("on my way", 30)]
        current = [message("on my way", 10), message("care mid", 30)]

        self.assertEqual(visible_overlap(previous, current), 1)
        self.assertEqual(
            [item.text for item in new_visible_messages(previous, current)],
            ["care mid"],
        )

    def test_legitimate_repeated_message_is_not_collapsed(self):
        previous = [message("hello", 10), message("gg", 30)]
        current = [message("gg", 10), message("gg", 30)]

        self.assertEqual(visible_overlap(previous, current), 1)
        self.assertEqual(
            [item.text for item in new_visible_messages(previous, current)], ["gg"]
        )

    def test_small_ocr_variation_still_matches(self):
        previous = [message("group dragon now", 10)]
        current = [message("group drag0n now", 10), message("omw", 30)]

        self.assertEqual(visible_overlap(previous, current), 1)


class WindowSelectionTests(unittest.TestCase):
    def test_prefers_game_window_over_riot_client(self):
        self.assertIsNone(league_window_rank("Riot Client"))
        self.assertEqual(league_window_rank("League of Legends"), 2)
        self.assertEqual(league_window_rank("League of Legends (TM) Client"), 3)
        self.assertGreater(
            league_window_rank("League of Legends (TM) Client"),
            league_window_rank("League of Legends"),
        )


class PrivacyTests(unittest.TestCase):
    def test_anonymizes_roster_names_riot_ids_and_links(self):
        result = anonymize_text(
            "Alex is with OtherUser#NA1 at https://example.com/Alex",
            names=["Alex"],
        )

        self.assertEqual(result, "[PLAYER] is with [PLAYER] at [LINK]")

    def test_does_not_replace_name_inside_another_word(self):
        self.assertEqual(
            anonymize_text("that is annoying Ann", names=["Ann"]),
            "that is annoying [PLAYER]",
        )


class ExportTests(unittest.TestCase):
    def test_export_schema_order_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            session = raw / "M-20260715-test"
            session.mkdir(parents=True)
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "match_id": "M-20260715-test",
                        "collection_started": "2026-07-15T20:00:00+00:00",
                        "collection_finished": "2026-07-15T20:30:00+00:00",
                        "candidate_count": 3,
                        "review_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            records = [
                {
                    "candidate_order": 1,
                    "decision": "keep",
                    "corrected_text": "first",
                    "exclusion_reason": "",
                },
                {
                    "candidate_order": 2,
                    "decision": "drop",
                    "corrected_text": "",
                    "exclusion_reason": "system",
                },
                {
                    "candidate_order": 3,
                    "decision": "keep",
                    "corrected_text": "second",
                    "exclusion_reason": "",
                },
            ]
            with (session / "reviewed.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            export = root / "final_chat.csv"
            manifest = root / "manifest.json"
            audit = root / "audit.json"
            matches, messages = export_reviewed(
                raw_dir=raw,
                export_path=export,
                manifest_path=manifest,
                audit_path=audit,
            )

            self.assertEqual((matches, messages), (1, 2))
            with export.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                list(rows[0]), ["match_id", "message_order", "text", "label", "notes"]
            )
            self.assertEqual([row["message_order"] for row in rows], ["1", "2"])
            self.assertEqual([row["text"] for row in rows], ["first", "second"])
            self.assertEqual(rows[0]["label"], "")
            self.assertEqual(
                json.loads(audit.read_text(encoding="utf-8"))["excluded_count"], 1
            )


if __name__ == "__main__":
    unittest.main()
