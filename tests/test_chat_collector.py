import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from collect_chat import (
    Calibration,
    ChatMessage,
    OCRLine,
    PENDING_MISS_LIMIT,
    ReconcileState,
    align_visible_messages,
    anonymize_text,
    can_continue_message,
    corrected_text_is_safe,
    ensure_calibration_window,
    export_reviewed,
    extract_chat_messages,
    image_digest,
    is_discarded,
    league_window_rank,
    looks_system_generated,
    new_visible_messages,
    preprocess_for_ocr,
    reconcile_visible,
    review_candidate,
    visible_overlap,
    _regroup_words_by_row,
)


def line(text: str, top: int, confidence: float = 90.0, height: int = 16) -> OCRLine:
    return OCRLine(
        text=text,
        confidence=confidence,
        left=10,
        top=top,
        width=300,
        height=height,
    )


def message(
    text: str,
    top: int,
    *,
    channel: str = "unknown",
    kind: str = "player",
    height: int = 16,
    confidence: float = 90.0,
) -> ChatMessage:
    return ChatMessage(
        text=text,
        channel=channel,
        confidence=confidence,
        kind=kind,
        left=10,
        top=top,
        width=200,
        height=height,
    )


class ParsingTests(unittest.TestCase):
    def test_parses_prefix_channel_and_wrapped_line(self):
        messages, stats = extract_chat_messages(
            [
                line("[12:34] [All] Player One (Lux): this is a", 10),
                line("wrapped message", 28),
                line("Player Two (Garen): gg", 46),
            ],
            roi_width=320,
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].channel, "all")
        self.assertEqual(messages[0].text, "this is a wrapped message")
        self.assertEqual(messages[0].kind, "player")
        self.assertEqual(messages[1].text, "gg")
        self.assertEqual(stats.player, 2)

    def test_filters_unstructured_noise(self):
        messages, stats = extract_chat_messages([line(". A oY. 1 .", 10)])

        self.assertEqual(messages, [])
        self.assertGreaterEqual(stats.noise, 1)

    def test_flags_system_notification(self):
        messages, stats = extract_chat_messages(
            [line("Blue turret destroyed", 10)]
        )
        self.assertEqual(messages, [])
        self.assertEqual(stats.system, 1)
        self.assertTrue(looks_system_generated("Blue turret destroyed"))

    def test_flags_ui_ping_and_objective_notifications(self):
        samples = [
            "Zaahen used Flash",
            "Akshan Flash",
            "Akshan. Flash",
            "Ahri Flash 'A",
            "1089 Gold",
            "| 13 Gold",
            "Master Yi - Alive",
            "- Alive",
            "Seraphine - Alive Vy",
            "Ocean Drake - Spawning in 1:30",
            "Baron Nashor - Spawning in 0:12",
            "Blitzcrank - Respawning in 30s",
            "Jungle Quest Complete!",
            "Guardian Angel - Need 459 gold",
            "Lich Bane - Need | gold",
            "Objective Bounties are available soon",
            "Smite - Ready - 1000 True Damage",
            "Wait For Blitzcrank Flash - 2s .",
            "E - Not learned yet",
            "Mana 22%",
            "Ahri used R - Spirit Rush",
            "[PLAYER] used R - Keeper's Verdict",
            "Master Yi R",
            "Akshan R",
            "Zyra R",
        ]
        for text in samples:
            with self.subTest(text=text):
                messages, stats = extract_chat_messages([line(text, 10)])
                self.assertEqual(messages, [])
                self.assertGreaterEqual(stats.system, 1)
                self.assertTrue(looks_system_generated(text))

        # Prefixed bodies that are only UI pings must not become candidates.
        messages, stats = extract_chat_messages(
            [
                line("[Team] Player (Lux): Zaahen used Flash", 10),
                line("Enemy (Akshan): 2112 Gold", 30),
                line("Enemy (Ahri): Mana 22%", 50),
                line("Ally (Seraphine): Ahri used R - Spirit Rush", 70),
            ]
        )
        self.assertEqual(messages, [])
        self.assertGreaterEqual(stats.system, 4)

        # Typed chat about spells/objectives must still pass.
        keep, keep_stats = extract_chat_messages(
            [
                line("[Team] Player (Lux): no flash", 10),
                line("[All] Player (Garen): go dragon", 30),
                line("[Team] Player (Zed): no r", 50),
            ]
        )
        self.assertEqual(
            [item.text for item in keep], ["no flash", "go dragon", "no r"]
        )
        self.assertEqual(keep_stats.player, 3)

    def test_scrubs_system_tail_glued_onto_player_chat(self):
        messages, stats = extract_chat_messages(
            [
                line(
                    "[Team] Player (Viego): go full ap Jeff said sorry. "
                    "You can tell them not to worry by pressing [I].",
                    10,
                ),
                line(
                    "[Team] Player (Lux): its funny Player (Akshan) has revived Player (Ahri)",
                    30,
                ),
                line(
                    "[Team] Player (Zed): i beamed that guy Player (Rengar) "
                    "1s asking for assistance",
                    50,
                ),
            ]
        )
        self.assertEqual(
            [item.text for item in messages],
            ["go full ap", "its funny", "i beamed that guy"],
        )
        self.assertEqual(stats.player, 3)

    def test_system_line_does_not_join_previous_player_message(self):
        messages, stats = extract_chat_messages(
            [
                line("[Team] Player (Viego): right clicker", 10),
                line("Jungle Quest Complete!", 28),
                line("[Team] Player (Lux): beamed that guy", 50),
                line("Player (Rengar) 1s asking for assistance", 68),
            ],
            roi_width=320,
        )
        self.assertEqual(
            [item.text for item in messages],
            ["right clicker", "beamed that guy"],
        )
        self.assertGreaterEqual(stats.system, 2)

    def test_flags_ping_and_client_only_notifications(self):
        messages, stats = extract_chat_messages(
            [
                line("Player (Lux) signals enemy has vision here", 10),
                line("Diana has selected Role: Mid", 30),
                line("Type /help for a list of commands", 50),
                line("[To] Friend#NA1: private message", 70),
                line("Jax is asking for assistance", 90),
                line("Soraka is on the way", 110),
            ]
        )
        self.assertEqual(messages, [])
        self.assertGreaterEqual(stats.system, 6)

    def test_malformed_prefixed_line_is_not_joined_to_previous_message(self):
        messages, _stats = extract_chat_messages(
            [
                line("[Team] Player (Lux): first", 10),
                line("[Team] OCR damaged second line", 30),
            ],
            roi_width=320,
        )
        self.assertEqual([item.text for item in messages], ["first"])

    def test_fallback_channel_prefix_becomes_uncertain(self):
        messages, stats = extract_chat_messages(
            [line("[All] Player One: come", 10)]
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].kind, "uncertain")
        self.assertEqual(messages[0].text, "come")
        self.assertEqual(stats.uncertain, 1)

    def test_continuation_requires_wide_previous_row(self):
        previous = message("this is a", 10)
        previous = ChatMessage(
            text=previous.text,
            channel="all",
            confidence=90,
            kind="player",
            left=10,
            top=10,
            width=100,
            height=16,
        )
        self.assertFalse(
            can_continue_message(previous, line("wrapped", 28), roi_width=320)
        )


class ImagePipelineTests(unittest.TestCase):
    def test_synthetic_screenshot_preprocessing(self):
        image = np.zeros((80, 320, 4), dtype=np.uint8)
        image[:, :, 3] = 255
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
        self.assertGreater(float(processed.mean()), 200.0)

    def test_digest_ignores_dark_terrain_changes_behind_same_text(self):
        first = np.full((80, 320, 4), (70, 80, 75, 255), dtype=np.uint8)
        second = np.full((80, 320, 4), (95, 85, 90, 255), dtype=np.uint8)
        for image in (first, second):
            cv2.putText(
                image,
                "Player: hello",
                (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        self.assertEqual(image_digest(first, cv2), image_digest(second, cv2))

    def test_geometry_regrouping_splits_wrong_tesseract_line_key(self):
        words = [
            ("[All]", 90.0, 10, 10, 40, 12),
            ("Player", 90.0, 55, 10, 50, 12),
            ("hello", 90.0, 10, 40, 40, 12),
        ]
        rows = _regroup_words_by_row(words)
        self.assertEqual(len(rows), 2)
        self.assertEqual([word[0] for word in rows[0]], ["[All]", "Player"])
        self.assertEqual([word[0] for word in rows[1]], ["hello"])


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

    def test_internal_ocr_drift_does_not_requeue_window(self):
        previous = [message(f"line {index}", 10 + index * 20) for index in range(8)]
        current = [
            message("line 0", 10),
            message("line 1x", 30),
            *[message(f"line {index}", 10 + index * 20) for index in range(2, 8)],
            message("new bottom", 170),
        ]
        added = new_visible_messages(previous, current)
        self.assertEqual([item.text for item in added], ["new bottom"])

    def test_zero_anchor_does_not_emit_all_new(self):
        previous = [message("alpha", 10), message("beta", 30)]
        current = [message("zzzz", 10), message("yyyy", 30)]
        self.assertEqual(new_visible_messages(previous, current), [])

    def test_stateful_confirmation_requires_second_sighting(self):
        state = ReconcileState(visible=[], pending=[], started_monotonic=1.0)
        first = reconcile_visible(state, [message("hello", 10)])
        self.assertEqual(first.confirmed, [])
        # Trailing uncertain lines still require a second sighting.
        second = reconcile_visible(
            first.state,
            [
                message("hello", 10),
                message("world", 30, kind="uncertain", confidence=70.0),
            ],
        )
        self.assertEqual([item.text for item in second.confirmed], ["hello"])
        third = reconcile_visible(
            second.state,
            [
                message("hello", 10),
                message("world", 30, kind="uncertain", confidence=70.0),
            ],
        )
        self.assertEqual([item.text for item in third.confirmed], ["world"])

    def test_high_confidence_player_emits_on_first_trailing_sighting(self):
        state = ReconcileState(visible=[], pending=[], started_monotonic=1.0)
        first = reconcile_visible(state, [message("hello", 10)])
        second = reconcile_visible(
            first.state, [message("hello", 10), message("gg", 30)]
        )
        self.assertEqual([item.text for item in second.confirmed], ["hello", "gg"])
        self.assertEqual(second.state.pending, [])

    def test_pending_survives_missed_frames_then_flushes(self):
        state = ReconcileState(visible=[], pending=[], started_monotonic=1.0)
        first = reconcile_visible(state, [message("hello", 10)])
        # Stage a low-confidence player line that is not eager-emitted.
        second = reconcile_visible(
            first.state,
            [
                message("hello", 10),
                message("care", 30, confidence=60.0),
            ],
        )
        self.assertEqual([item.text for item in second.confirmed], ["hello"])
        self.assertEqual(len(second.state.pending), 1)
        # Missed frames age pending instead of dropping immediately.
        aged = second.state
        for _ in range(PENDING_MISS_LIMIT + 1):
            aged = reconcile_visible(aged, [], parse_ok=False).state
            if not aged.pending:
                break
        # Low-confidence pending expires as an unconfirmed drop.
        self.assertEqual(aged.pending, [])
        self.assertGreaterEqual(aged.unconfirmed_drops, 1)

    def test_high_confidence_pending_flushes_after_miss_limit(self):
        state = ReconcileState(visible=[], pending=[], started_monotonic=1.0)
        first = reconcile_visible(state, [message("hello", 10)])
        # Baseline stages without eager emit; fading chat should still flush.
        flushed = []
        current_state = first.state
        for _ in range(4):
            result = reconcile_visible(current_state, [], parse_ok=False)
            flushed.extend(result.confirmed)
            current_state = result.state
            if not current_state.pending:
                break
        self.assertEqual([item.text for item in flushed], ["hello"])
        self.assertEqual(current_state.pending, [])

    def test_alignment_maps_repeated_gg_to_earliest_occurrence(self):
        previous = [message("hello", 10), message("gg", 30)]
        current = [message("gg", 10), message("gg", 30)]
        alignment = align_visible_messages(previous, current)
        self.assertEqual(alignment.trailing_indices, [1])


class WindowSelectionTests(unittest.TestCase):
    def test_prefers_game_window_over_riot_client(self):
        self.assertIsNone(league_window_rank("Riot Client"))
        self.assertEqual(league_window_rank("League of Legends"), 2)
        self.assertEqual(league_window_rank("League of Legends (TM) Client"), 3)
        self.assertGreater(
            league_window_rank("League of Legends (TM) Client"),
            league_window_rank("League of Legends"),
        )

    def test_rejects_different_window_from_calibration(self):
        calibration = Calibration(
            window_title="League of Legends (TM) Client",
            x_ratio=0.0,
            y_ratio=0.5,
            width_ratio=0.4,
            height_ratio=0.4,
            created_at="2026-08-01T00:00:00+00:00",
            active_match_verified=True,
        )
        with self.assertRaises(RuntimeError):
            ensure_calibration_window(calibration, "League of Legends")


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

    def test_anonymizes_ocr_damaged_double_hash_riot_id(self):
        self.assertEqual(
            anonymize_text("Lunchi##333 said hello"),
            "[PLAYER] said hello",
        )

    def test_rejects_prefix_shaped_corrected_text(self):
        self.assertFalse(
            corrected_text_is_safe("[All] Player One (Lux): hello")
        )
        self.assertTrue(corrected_text_is_safe("hello"))

    def test_review_requires_explicit_action_for_uncertain(self):
        record = {
            "candidate_order": 1,
            "kind": "uncertain",
            "ocr_confidence": 50.0,
            "text": "maybe",
            "screenshot_ref": "raw/x/screenshots/1.png",
        }
        answers = iter(["", "s"])
        decision, text, reason = review_candidate(
            record, input_fn=lambda _prompt: next(answers)
        )
        self.assertEqual((decision, text, reason), ("drop", "", "system"))

    def test_review_drop_shortcuts_and_repeat_last_reason(self):
        record = {
            "candidate_order": 2,
            "kind": "player",
            "ocr_confidence": 90.0,
            "text": "gg",
            "screenshot_ref": "raw/x/screenshots/2.png",
        }
        decision, text, reason = review_candidate(
            record, input_fn=lambda _prompt: "o"
        )
        self.assertEqual((decision, text, reason), ("drop", "", "ocr duplicate"))

        # After an OCR-dup drop, bare [d]+Enter repeats that reason.
        answers = iter(["d", ""])
        decision, text, reason = review_candidate(
            record, input_fn=lambda _prompt: next(answers)
        )
        self.assertEqual((decision, text, reason), ("drop", "", "ocr duplicate"))

        decision, text, reason = review_candidate(
            record, input_fn=lambda _prompt: "s"
        )
        self.assertEqual((decision, text, reason), ("drop", "", "system"))


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
                        "discarded": False,
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

    def test_discarded_sessions_are_skipped_by_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            session = raw / "M-discard"
            session.mkdir(parents=True)
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "match_id": "M-discard",
                        "collection_started": "2026-08-01T00:00:00+00:00",
                        "collection_finished": "2026-08-01T00:10:00+00:00",
                        "candidate_count": 10,
                        "review_complete": False,
                        "discarded": True,
                        "discard_reason": "collector_pipeline_failure",
                    }
                ),
                encoding="utf-8",
            )
            matches, messages = export_reviewed(
                raw_dir=raw,
                export_path=root / "final_chat.csv",
                manifest_path=root / "manifest.json",
                audit_path=root / "audit.json",
            )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((matches, messages), (0, 0))
            self.assertEqual(manifest["discarded_matches"], 1)
            self.assertTrue(is_discarded(json.loads((session / "session.json").read_text(encoding="utf-8"))))


if __name__ == "__main__":
    unittest.main()
