"""Passive screen-OCR collector for the fresh League of Legends final test.

The Riot APIs do not expose in-game team/all chat. This tool therefore reads
only pixels already visible to the user. It never injects input, reads process
memory, or assigns toxicity labels.

Typical use:

    python scripts/collect_chat.py calibrate
    python scripts/collect_chat.py collect
    python scripts/collect_chat.py review
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_DIR = ROOT / "data" / "final_test"
RAW_DIR = FINAL_TEST_DIR / "raw"
CALIBRATION_PATH = RAW_DIR / "calibration.json"
EXPORT_PATH = FINAL_TEST_DIR / "final_chat.csv"
MANIFEST_PATH = FINAL_TEST_DIR / "collection_manifest.json"
AUDIT_PATH = FINAL_TEST_DIR / "exclusion_audit.json"

LIVE_API = "https://127.0.0.1:2999/liveclientdata"
OCR_SCALE = 3.0
OCR_BORDER = 16
MIN_OCR_CONFIDENCE = 20.0
MIN_STRUCTURE_CONFIDENCE = 5.0
ROW_CENTER_TOLERANCE_PX = 8
MAX_ROW_HEIGHT_FACTOR = 2.5
MATCH_RATIO = 0.82
SHORT_TEXT_LEN = 3
VERTICAL_RESIDUAL_PX = 6
RATE_LIMIT_PER_MINUTE = 30
SUSPICIOUS_SUFFIX = 4
# High-confidence champion-prefixed lines are emitted on first trailing sighting
# so chat fade / OCR miss before a second poll does not drop real player chat.
EAGER_EMIT_CONFIDENCE = 85.0
# Allow a few missed frames before dropping unconfirmed pending tracks.
PENDING_MISS_LIMIT = 3
TARGET_MIN_MESSAGES = 200
TARGET_MAX_MESSAGES = 300
TARGET_MATCHES = 20
CALIBRATION_ROI_PATH = RAW_DIR / "calibration_roi.png"
CALIBRATION_OCR_PATH = RAW_DIR / "calibration_ocr_input.png"
PLAYER_PREFIX_SHAPE_RE = re.compile(
    r"(?i)^\s*(?:\[\s*(?:all|team|party)\s*\]\s*)?"
    r".{1,64}\s+\([^)]+\)\s*:"
)

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
RIOT_ID_RE = re.compile(
    r"(?<!\w)[A-Za-z0-9][A-Za-z0-9._'-]{1,22}\s*#+\s*[A-Za-z0-9]{3,6}(?!\w)"
)
TIMESTAMP_RE = r"(?:\[\s*\d{1,2}:\d{2}\s*\]\s*)?"
# Strong player lines require a champion parenthesis and a body separator.
CHAT_PREFIX_RE = re.compile(
    rf"^\s*{TIMESTAMP_RE}"
    r"(?:\[\s*(?P<channel>all|team|party)\s*\]\s*)?"
    r"(?P<speaker>[^:\r\n()]{1,64}?)"
    r"\s+\((?P<champion>[^)]+)\)\s*[:;\-–—]\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)
FALLBACK_PREFIX_RE = re.compile(
    rf"^\s*{TIMESTAMP_RE}"
    r"\[\s*(?P<channel>all|team|party)\s*\]\s*"
    r"(?P<speaker>[^:\r\n()]{1,64}?)"
    r"(?:\s+\((?P<champion>[^)]+)\))?\s*[:;\-–—]\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)
SUMMONER_SPELLS = (
    r"Flash|Heal|Teleport|Ignite|Exhaust|Barrier|Cleanse|Ghost|Smite|"
    r"Clarity|Mark|Dash"
)
# Shared champion-like name token(s) used by cooldown / ability pings.
_CHAMP_NAME = (
    r"(?!(?:no|my|his|her|their|enemy|your|the|has|have|use|bait|"
    r"save|ward|with|our|its|go|not|wait)\b)"
    r"[A-Za-z][A-Za-z.''\-]{1,20}"
    r"(?:\s+[A-Za-z][A-Za-z.''\-]{1,20}){0,2}"
)
# Cooldown pings are usually "Champion Spell" / "Champion. Spell", not chat
# about flash ("no flash", "enemy flash"). Allow a little trailing OCR junk.
SUMMONER_SPELL_PING_RE = (
    rf"^\s*{_CHAMP_NAME}"
    rf"\s*[.\u2022·]?\s*(?:{SUMMONER_SPELLS})\b"
    rf"(?:\s*\S{{0,8}})?\s*$"
)
# Ability cooldown pings ("Master Yi R", "Akshan R", "Zyra R").
ABILITY_COOLDOWN_PING_RE = rf"^\s*{_CHAMP_NAME}\s+[QWER]\b(?:\s*\S{{0,8}})?\s*$"
# OCR often reads "is" as "1s"/"ts"/"ls" in smart-ping templates.
_OCR_IS = r"(?:is|1s|ts|ls|Is)"
SYSTEM_LINE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bhas (?:slain|destroyed|joined|left|disconnected|reconnected|revived)\b",
        r"\bhas shut down\b",
        r"\bis on a (?:killing spree|rampage)\b",
        rf"\b{_OCR_IS}\s+(?:dominating|godlike|legendary|asking for assistance|on the way)\b",
        r"\b(?:shutdown|first blood|ace!?)\b",
        r"\b(?:enemy|ally) (?:slain|missing)\b",
        r"\bpurchased\b",
        r"\b(?:turret|inhibitor) (?:destroyed|respawned)\b",
        r"\bsignals (?:that )?.+\b",
        r"\bhas selected role\b",
        r"^\s*type /help\b",
        r"^\s*\[(?:to|from)\]\s*",
        r"\bsurrender(?:ed|s)?\b",
        r"\btransformed into\b",
        r"\bis available\b",
        # Scoreboard / tab-target pings ("Master Yi - Alive", with OCR junk).
        r"(?:^|-\s*)Alive\b",
        r"\bRespawning in\b",
        r"\bporo[- ]?snax\b",
        r"\bcharges\b",
        r"\b(?:double|triple|quadra|penta) kill\b",
        r"\bphenomenal evil\b",
        r"\bcourage of the colossus\b",
        r"\bzealot\b",
        r"\bdropkick\b",
        r"\bfinal form\b",
        r"\bwait for r\b",
        # "Wait For Blitzcrank Flash - 2s"
        rf"\bwait for\b.+\b(?:{SUMMONER_SPELLS}|[qwer])\b",
        r"\bnot learned yet\b",
        r"\battack speed\b",
        r"\b(?:guinsoo|runaan|rageblade|hurricane)\b",
        # Summoner-spell use and cooldown pings.
        rf"\bused\s+(?:{SUMMONER_SPELLS})\b",
        SUMMONER_SPELL_PING_RE,
        rf"^\s*(?:{SUMMONER_SPELLS})\s*$",
        # Ultimate / ability cast announcements ("Ahri used R - Spirit Rush").
        r"\bused\s+[QWER]\s*-",
        r"\bused\s+(?:ult|ultimate)\b",
        ABILITY_COOLDOWN_PING_RE,
        # Smite (and similar) readiness tooltips in chat.
        rf"^\s*(?:{SUMMONER_SPELLS})\s*-\s*Ready\b",
        r"\bTrue Damage\b",
        # Resource pings.
        r"^\s*(?:Mana|Energy)\s+\d{1,3}%\s*$",
        # Gold / shop pings (OCR may turn "1" into "|" / "l" / "I").
        r"^\s*[|Il]?\s*\d{1,5}\s*Gold\s*$",
        r"\bNeed\s+[\d|Il]+\s+go(?:ld)?\b",
        # Objective timers and announcements (avoid bare "dragon"/"herald" chat).
        r"\bSpawning in\b",
        r"^\s*(?:Baron\s+Nashor|(?:Ocean|Mountain|Infernal|Cloud|Hextech|Chemtech)\s+Drake|"
        r"Elder\s+Dragon|Atakhan|Voidgrubs?|Rift Herald)\b",
        r"\bObjective Bounties\b",
        r"\b(?:Jungle\s+)?Quest Complete\b",
        # Smart-ping / honor templates that OCR into chat.
        r"\bwants to (?:push forward|retreat|attack|defend|group)\b",
        r"said sorry\b",
        r"not to worry by pressing\b",
    )
)
NON_PLAYER_ACTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"^\s*(?:{_OCR_IS}\s+)?(?:on the way|asking for assistance)\s*$",
        r"^\s*(?:leona\s+)?dropkick\s*$",
        r"^\s*.+\s+(?:zealot|phenomenal evil|courage of the colossus|final form)\s*$",
        r"^\s*.*poro[- ]?snax.*$",
        r"^\s*.*\bcharges\b.*$",
        r"^\s*wait for r\b",
        rf"^\s*wait for\b.+\b(?:{SUMMONER_SPELLS}|[qwer])\b",
        r"^\s*attack speed\b",
        r"\b(?:guinsoo|runaan|rageblade|hurricane|infinity edge|kraken)\b",
        SUMMONER_SPELL_PING_RE,
        ABILITY_COOLDOWN_PING_RE,
        rf"\bused\s+(?:{SUMMONER_SPELLS})\b",
        r"\bused\s+[QWER]\s*-",
        rf"^\s*(?:{SUMMONER_SPELLS})\s*-\s*Ready\b",
        r"\bTrue Damage\b",
        r"^\s*(?:Mana|Energy)\s+\d{1,3}%\s*$",
        r"^\s*[|Il]?\s*\d{1,5}\s*Gold\s*$",
        r"(?:^|-\s*)Alive\b",
        r"\bRespawning in\b",
        r"\bSpawning in\b",
        r"\b(?:Jungle\s+)?Quest Complete\b",
        r"\bNeed\s+[\d|Il]+\s+go(?:ld)?\b",
        r"\bObjective Bounties\b",
        r"\bwants to (?:push forward|retreat|attack|defend|group)\b",
        r"\bnot learned yet\b",
        r"\bhas revived\b",
        r"said sorry\b",
        r"not to worry by pressing\b",
    )
)
# When a system ping is glued onto a real chat body, keep only the chat prefix.
# Speaker tokens must start with a real capital letter (case-sensitive) so typed
# lowercase words are not eaten by the splitter.
_SPEAKER_NAME = (
    r"(?:\[PLAYER\]|(?-i:[A-Z])[A-Za-z0-9.''\-]{1,20}"
    r"(?:\s+(?-i:[A-Z])[A-Za-z0-9.''\-]{1,20}){0,2})"
)
SYSTEM_TAIL_SPLIT_RE = re.compile(
    rf"\s+(?:"
    rf"{_SPEAKER_NAME}\s+\([^)]+\)\s+"
    rf"(?:{_OCR_IS}\s+)?(?:asking for assistance|on the way)\b|"
    rf"{_SPEAKER_NAME}\s+\([^)]+\)\s+has revived\b|"
    rf"{_SPEAKER_NAME}\s+\([^)]+\)\s+purchased\b|"
    rf"{_SPEAKER_NAME}\s+used\s+[QWER]\s*-|"
    rf"{_SPEAKER_NAME}\s+said sorry\b|"
    r"said sorry\b|"
    r"not to worry by pressing\b|"
    r"You can tell them not to worry\b"
    r").*$",
    re.IGNORECASE,
)
CHAT_START_RE = re.compile(
    rf"^\s*{TIMESTAMP_RE}(?:\[\s*(?:all|team|party|to|from)\s*\]|"
    r"[^:\r\n]{1,64}\s+\([^)]+\)\s*[:;\-–—])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Calibration:
    """Chat rectangle expressed relative to the League client area."""

    window_title: str
    x_ratio: float
    y_ratio: float
    width_ratio: float
    height_ratio: float
    created_at: str
    active_match_verified: bool = False


@dataclass(frozen=True)
class OCRLine:
    """One visual line returned by OCR."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class ChatMessage:
    """A parsed visible chat message, potentially spanning OCR lines."""

    text: str
    channel: str
    confidence: float
    kind: str
    left: int
    top: int
    width: int
    height: int


@dataclass
class ParseStats:
    player: int = 0
    uncertain: int = 0
    system: int = 0
    noise: int = 0


@dataclass
class AlignmentResult:
    matched_pairs: list[tuple[int, int]]
    trailing_indices: list[int]
    shift: float
    confidence: str
    residual: float


@dataclass
class TrackedMessage:
    track_id: int
    message: ChatMessage
    sightings: int = 1
    emitted: bool = False
    pending: bool = True
    first_seen_order: int = 0
    misses: int = 0


@dataclass
class ReconcileState:
    visible: list[TrackedMessage]
    pending: list[TrackedMessage]
    resync_snapshot: list[ChatMessage] | None = None
    next_track_id: int = 1
    unanchored_frames: int = 0
    resyncs: int = 0
    bulk_suppressed: int = 0
    unconfirmed_drops: int = 0
    emitted_count: int = 0
    started_monotonic: float = 0.0
    failed: bool = False
    failure_reason: str = ""
    last_emitted_norm: str = ""


@dataclass
class ReconcileResult:
    confirmed: list[ChatMessage]
    state: ReconcileState
    batch_size: int = 0
    bad_frame: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Chat capture is currently supported only on Windows.")


def enable_dpi_awareness() -> None:
    """Make Win32 window rects match physical pixels used by mss.

    Without this, displays with scaling (125%/150%/…) return logical coordinates
    and the capture region can land on the wrong app (often the IDE).
    """

    if sys.platform != "win32":
        return
    try:
        import ctypes

        # Per-monitor DPI awareness v2 (Windows 10 1703+).
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def import_capture_dependencies():
    """Import optional capture packages only for commands that need them."""

    try:
        import cv2  # type: ignore
        import mss  # type: ignore
        import numpy as np  # type: ignore
        import pytesseract  # type: ignore
        import win32gui  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing capture dependency. Activate .venv and run "
            "`pip install -r requirements.txt`."
        ) from exc
    return cv2, mss, np, pytesseract, win32gui


def ensure_tesseract_available(pytesseract) -> None:
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not on PATH. Install it as "
            "documented in README.md, or pass --tesseract-cmd."
        ) from exc


def league_window_rank(title: str) -> int | None:
    """Return a preference rank for a window title, or None if not a game window.

    Higher is better. The Riot launcher is excluded. Prefer the common in-game
    title "League of Legends (TM) Client" when several League windows exist.
    """

    lowered = title.strip().lower()
    if not lowered or lowered == "riot client":
        return None
    if "riot client" in lowered and "league of legends" not in lowered:
        return None
    if "league of legends" not in lowered:
        return None
    if "(tm) client" in lowered:
        return 3
    if lowered == "league of legends":
        return 2
    return 1


def find_league_window(win32gui, hwnd: int | None = None) -> tuple[int, str]:
    """Return the preferred visible League game window."""

    if hwnd is not None:
        if (
            not win32gui.IsWindow(hwnd)
            or not win32gui.IsWindowVisible(hwnd)
            or win32gui.IsIconic(hwnd)
        ):
            raise RuntimeError(f"Window handle {hwnd} is not a visible window.")
        title = win32gui.GetWindowText(hwnd).strip() or f"hwnd:{hwnd}"
        if league_window_rank(title) is None:
            raise RuntimeError(
                f"Window handle {hwnd} is not a League window (title: {title!r})."
            )
        return hwnd, title

    # (rank, area, hwnd, title)
    candidates: list[tuple[int, int, int, str]] = []

    def visit(candidate_hwnd: int, _extra: object) -> None:
        if (
            not win32gui.IsWindowVisible(candidate_hwnd)
            or win32gui.IsIconic(candidate_hwnd)
        ):
            return
        title = win32gui.GetWindowText(candidate_hwnd).strip()
        rank = league_window_rank(title)
        if rank is None:
            return
        left, top, right, bottom = win32gui.GetWindowRect(candidate_hwnd)
        area = max(0, right - left) * max(0, bottom - top)
        if area:
            candidates.append((rank, area, candidate_hwnd, title))

    win32gui.EnumWindows(visit, None)
    if not candidates:
        raise RuntimeError(
            "League game window not found. Start a Practice Tool/custom game "
            "in borderless or windowed mode, then run calibrate again. "
            "Use `python scripts/collect_chat.py list-windows` to inspect titles."
        )
    _rank, _area, selected_hwnd, title = max(candidates)
    return selected_hwnd, title


def list_capture_windows(win32gui) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    """Return visible top-level windows with non-empty titles for debugging."""

    windows: list[tuple[int, str, tuple[int, int, int, int]]] = []

    def visit(hwnd: int, _extra: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        windows.append((hwnd, title, (left, top, right - left, bottom - top)))

    win32gui.EnumWindows(visit, None)
    windows.sort(key=lambda item: item[2][2] * item[2][3], reverse=True)
    return windows


def client_rect(win32gui, hwnd: int) -> tuple[int, int, int, int]:
    """Return client-area coordinates as absolute x, y, width, height."""

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
    screen_right, screen_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return (
        screen_left,
        screen_top,
        screen_right - screen_left,
        screen_bottom - screen_top,
    )


def grab_region(sct, np, rect: tuple[int, int, int, int]):
    x, y, width, height = rect
    shot = sct.grab({"left": x, "top": y, "width": width, "height": height})
    return np.asarray(shot)


def open_screen_capture(mss):
    """Return an mss screen-capture context (MSS preferred over deprecated mss())."""

    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


def save_calibration(calibration: Calibration) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps(asdict(calibration), indent=2), encoding="utf-8"
    )


def load_calibration() -> Calibration:
    if not CALIBRATION_PATH.exists():
        raise RuntimeError("No calibration found. Run `python scripts/collect_chat.py calibrate`.")
    try:
        calibration = Calibration(
            **json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Calibration is invalid. Run `python scripts/collect_chat.py calibrate` again."
        ) from exc
    ratios = (
        calibration.x_ratio,
        calibration.y_ratio,
        calibration.width_ratio,
        calibration.height_ratio,
    )
    if (
        any(
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
            for ratio in ratios
        )
        or calibration.width_ratio <= 0
        or calibration.height_ratio <= 0
        or calibration.x_ratio + calibration.width_ratio > 1.001
        or calibration.y_ratio + calibration.height_ratio > 1.001
    ):
        raise RuntimeError(
            "Calibration rectangle is outside the League window. Recalibrate."
        )
    return calibration


def calibrated_rect(
    calibration: Calibration, window_rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x, y, width, height = window_rect
    return (
        x + round(width * calibration.x_ratio),
        y + round(height * calibration.y_ratio),
        max(1, round(width * calibration.width_ratio)),
        max(1, round(height * calibration.height_ratio)),
    )


def ensure_calibration_window(calibration: Calibration, title: str) -> None:
    """Reject applying saved ratios to a different League window type."""

    if calibration.window_title.strip().casefold() != title.strip().casefold():
        raise RuntimeError(
            "The active League window differs from the calibrated window "
            f"({title!r} != {calibration.window_title!r}). Recalibrate in-game."
        )


def _to_bgr(image, cv2):
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _pad_ocr_page(binary, cv2):
    return cv2.copyMakeBorder(
        binary,
        OCR_BORDER,
        OCR_BORDER,
        OCR_BORDER,
        OCR_BORDER,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _chat_glyph_mask(bgr, cv2):
    """Isolate LoL chat glyphs from translucent terrain/UI behind the panel.

    White message bodies are high-value/low-saturation; names and system lines
    are saturated bright UI colors. Terrain tends to be mid-value and is dropped.
    """

    import numpy as np

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)
    white = ((value >= 175) & (saturation <= 90)).astype(np.uint8) * 255
    colored = ((value >= 150) & (saturation >= 45)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(white, colored)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)
    height, width = mask.shape
    max_area = max(64, int(0.08 * height * width))
    for index in range(1, component_count):
        _x, _y, comp_width, comp_height, area = stats[index]
        if area < 12 or area > max_area:
            continue
        if comp_width > 0.55 * width and comp_height > 0.2 * height:
            continue
        if comp_height > 0.35 * height:
            continue
        clean[labels == index] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)


def preprocess_for_ocr(image, cv2):
    """Upscale colored LoL chat into dark text on a light background for Tesseract."""

    enlarged = cv2.resize(
        _to_bgr(image, cv2),
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    mask = _chat_glyph_mask(enlarged, cv2)
    binary = cv2.bitwise_not(mask)
    return _pad_ocr_page(binary, cv2)


def _collect_ocr_words(data) -> list[tuple[str, float, int, int, int, int]]:
    words: list[tuple[str, float, int, int, int, int]] = []
    for index, raw in enumerate(data["text"]):
        word = str(raw).strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if not word or confidence < MIN_STRUCTURE_CONFIDENCE:
            continue
        words.append(
            (
                word,
                confidence,
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
            )
        )
    return words


def _regroup_words_by_row(
    words: Sequence[tuple[str, float, int, int, int, int]],
) -> list[list[tuple[str, float, int, int, int, int]]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda item: (item[3] + item[5] / 2.0, item[2]))
    rows: list[list[tuple[str, float, int, int, int, int]]] = [[ordered[0]]]
    current_center = ordered[0][3] + ordered[0][5] / 2.0
    for word in ordered[1:]:
        center = word[3] + word[5] / 2.0
        if abs(center - current_center) <= ROW_CENTER_TOLERANCE_PX * OCR_SCALE:
            rows[-1].append(word)
            current_center = sum(
                item[3] + item[5] / 2.0 for item in rows[-1]
            ) / len(rows[-1])
        else:
            rows.append([word])
            current_center = center
    return rows


def ocr_lines(image, cv2, pytesseract) -> list[OCRLine]:
    """Extract visual text lines by geometry, not unstable Tesseract line IDs."""

    processed = preprocess_for_ocr(image, cv2)
    data = pytesseract.image_to_data(
        processed,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )
    rows = _regroup_words_by_row(_collect_ocr_words(data))
    if not rows:
        return []

    median_height = sorted(max(1, word[5]) for row in rows for word in row)
    typical = median_height[len(median_height) // 2]
    max_height = max(typical * MAX_ROW_HEIGHT_FACTOR, typical + 4 * OCR_SCALE)

    lines: list[OCRLine] = []
    for row in rows:
        strong = [word for word in row if word[1] >= MIN_OCR_CONFIDENCE]
        usable = strong or row
        usable.sort(key=lambda item: item[2])
        left = min(item[2] for item in usable) - OCR_BORDER
        top = min(item[3] for item in usable) - OCR_BORDER
        right = max(item[2] + item[4] for item in usable) - OCR_BORDER
        bottom = max(item[3] + item[5] for item in usable) - OCR_BORDER
        height = bottom - top
        if height > max_height:
            continue
        text = " ".join(item[0] for item in usable).strip()
        if not text:
            continue
        confidence = sum(item[1] for item in usable) / len(usable)
        if confidence < MIN_OCR_CONFIDENCE and not CHAT_START_RE.match(text):
            continue
        lines.append(
            OCRLine(
                text=text,
                confidence=confidence,
                left=round(left / OCR_SCALE),
                top=round(top / OCR_SCALE),
                width=round((right - left) / OCR_SCALE),
                height=round(height / OCR_SCALE),
            )
        )
    return sorted(lines, key=lambda line: (line.top, line.left))


def looks_system_line(text: str) -> bool:
    return any(pattern.search(text) for pattern in SYSTEM_LINE_PATTERNS)


def looks_non_player_action(text: str) -> bool:
    return any(pattern.search(text) for pattern in NON_PLAYER_ACTION_PATTERNS)


def looks_system_generated(text: str) -> bool:
    """Backward-compatible alias used by older tests and call sites."""

    return looks_system_line(text) or looks_non_player_action(text)


def scrub_system_tail(text: str) -> str:
    """Drop UI/system fragments that OCR glued onto a typed chat body."""

    return SYSTEM_TAIL_SPLIT_RE.sub("", text).strip()


def looks_noise_text(text: str) -> bool:
    cleaned = re.sub(r"[^\w]+", "", text)
    if len(text.strip()) < 2:
        return True
    if len(cleaned) < 2:
        return True
    letters = sum(char.isalpha() for char in text)
    return letters / max(1, len(text.replace(" ", ""))) < 0.35


def can_continue_message(
    previous: ChatMessage, line: OCRLine, roi_width: int | None = None
) -> bool:
    if looks_system_line(line.text) or looks_non_player_action(line.text):
        return False
    if CHAT_START_RE.match(line.text) or CHAT_PREFIX_RE.match(line.text):
        return False
    if FALLBACK_PREFIX_RE.match(line.text):
        return False
    gap = line.top - (previous.top + previous.height)
    if gap < -4 or gap > max(12, previous.height):
        return False
    typical = max(8, previous.height)
    if line.height > typical * MAX_ROW_HEIGHT_FACTOR:
        return False
    if roi_width is not None and previous.left + previous.width < roi_width * 0.55:
        return False
    return True


def extract_chat_messages(
    lines: Sequence[OCRLine], roi_width: int | None = None
) -> tuple[list[ChatMessage], ParseStats]:
    """Parse chat-shaped lines and count filtered system/noise rows."""

    messages: list[ChatMessage] = []
    stats = ParseStats()
    for line in lines:
        full = line.text.strip()
        if not full:
            stats.noise += 1
            continue

        # Prefer chat-prefix parsing before full-line system checks so a real
        # message that OCR glued onto a UI ping can still be recovered.
        match = CHAT_PREFIX_RE.match(full)
        if match:
            text = scrub_system_tail(match.group("text").strip())
            if looks_non_player_action(text) or looks_system_line(text):
                stats.system += 1
                continue
            if not text or looks_noise_text(text):
                stats.noise += 1
                continue
            messages.append(
                ChatMessage(
                    text=text,
                    channel=(match.group("channel") or "unknown").lower(),
                    confidence=line.confidence,
                    kind="player",
                    left=line.left,
                    top=line.top,
                    width=line.width,
                    height=line.height,
                )
            )
            stats.player += 1
            continue

        fallback = FALLBACK_PREFIX_RE.match(full)
        if fallback and fallback.group("channel"):
            text = scrub_system_tail(fallback.group("text").strip())
            if looks_non_player_action(text) or looks_system_line(text):
                stats.system += 1
                continue
            if text and not looks_noise_text(text):
                messages.append(
                    ChatMessage(
                        text=text,
                        channel=fallback.group("channel").lower(),
                        confidence=line.confidence,
                        kind="uncertain",
                        left=line.left,
                        top=line.top,
                        width=line.width,
                        height=line.height,
                    )
                )
                stats.uncertain += 1
                continue
            stats.noise += 1
            continue

        if looks_system_line(full) or looks_non_player_action(full):
            stats.system += 1
            continue
        if looks_noise_text(full):
            stats.noise += 1
            continue

        if messages and can_continue_message(messages[-1], line, roi_width=roi_width):
            previous = messages[-1]
            combined = scrub_system_tail(f"{previous.text} {full}".strip())
            if not combined or looks_system_line(combined) or looks_non_player_action(
                combined
            ):
                stats.system += 1
                continue
            messages[-1] = ChatMessage(
                text=combined,
                channel=previous.channel,
                confidence=(previous.confidence + line.confidence) / 2,
                kind=previous.kind,
                left=min(previous.left, line.left),
                top=previous.top,
                width=max(previous.left + previous.width, line.left + line.width)
                - min(previous.left, line.left),
                height=max(previous.top + previous.height, line.top + line.height)
                - previous.top,
            )
            continue

        stats.noise += 1
    return messages, stats


def normalized_match_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def message_match_score(previous: ChatMessage, current: ChatMessage) -> float:
    first = normalized_match_text(previous.text)
    second = normalized_match_text(current.text)
    if not first or not second:
        return 0.0
    if len(first) <= SHORT_TEXT_LEN or len(second) <= SHORT_TEXT_LEN:
        return 1.0 if first == second else 0.0
    ratio = difflib.SequenceMatcher(None, first, second).ratio()
    if ratio < MATCH_RATIO:
        return 0.0
    bonus = 0.0
    if previous.channel == current.channel and previous.channel != "unknown":
        bonus += 0.02
    if previous.kind == current.kind:
        bonus += 0.01
    return min(1.0, ratio + bonus)


def messages_match(previous: ChatMessage, current: ChatMessage) -> bool:
    return message_match_score(previous, current) >= MATCH_RATIO


def _estimate_shift_candidates(
    previous: Sequence[ChatMessage], current: Sequence[ChatMessage]
) -> list[float]:
    offsets: list[float] = []
    for old in previous:
        for new in current:
            if message_match_score(old, new) <= 0:
                continue
            offsets.append(float(new.top - old.top))
    if not offsets:
        return [0.0]
    offsets.sort()
    clusters: list[list[float]] = [[offsets[0]]]
    for value in offsets[1:]:
        if abs(value - clusters[-1][-1]) <= VERTICAL_RESIDUAL_PX:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    medians = [sorted(cluster)[len(cluster) // 2] for cluster in clusters]
    if 0.0 not in medians:
        medians.append(0.0)
    return sorted(set(medians), key=lambda item: (item > 8, abs(item)))


def align_visible_messages(
    previous: Sequence[ChatMessage], current: Sequence[ChatMessage]
) -> AlignmentResult:
    """Geometry-aware ordered alignment that never treats zero anchors as all-new."""

    if not previous:
        return AlignmentResult(
            matched_pairs=[],
            trailing_indices=list(range(len(current))),
            shift=0.0,
            confidence="none" if not current else "low",
            residual=0.0,
        )
    if not current:
        return AlignmentResult([], [], 0.0, "none", 0.0)

    typical_height = max(
        8,
        round(sum(item.height for item in previous) / max(1, len(previous))),
    )
    residual_limit = max(VERTICAL_RESIDUAL_PX, 0.4 * typical_height)
    best: AlignmentResult | None = None

    for shift in _estimate_shift_candidates(previous, current):
        if shift > typical_height:
            continue
        scores = [
            [0.0 for _ in range(len(current) + 1)] for _ in range(len(previous) + 1)
        ]
        choices = [
            ["" for _ in range(len(current) + 1)] for _ in range(len(previous) + 1)
        ]
        for old_index, old in enumerate(previous, start=1):
            for new_index, new in enumerate(current, start=1):
                skip_old = scores[old_index - 1][new_index]
                skip_new = scores[old_index][new_index - 1]
                score = message_match_score(old, new)
                residual = abs((new.top - old.top) - shift)
                match = -1.0
                if score > 0 and residual <= residual_limit:
                    uniqueness = sum(
                        1
                        for other in current
                        if normalized_match_text(other.text)
                        == normalized_match_text(new.text)
                    )
                    match = (
                        scores[old_index - 1][new_index - 1]
                        + score
                        + (0.15 if uniqueness == 1 else 0.0)
                        + (len(current) - new_index) * 0.001
                        - residual / 100.0
                    )
                if match >= skip_old and match >= skip_new and match >= 0:
                    scores[old_index][new_index] = match
                    choices[old_index][new_index] = "match"
                elif skip_old >= skip_new:
                    scores[old_index][new_index] = skip_old
                    choices[old_index][new_index] = "skip_old"
                else:
                    scores[old_index][new_index] = skip_new
                    choices[old_index][new_index] = "skip_new"

        pairs: list[tuple[int, int]] = []
        old_index = len(previous)
        new_index = len(current)
        while old_index > 0 and new_index > 0:
            choice = choices[old_index][new_index]
            if choice == "match":
                pairs.append((old_index - 1, new_index - 1))
                old_index -= 1
                new_index -= 1
            elif choice == "skip_old":
                old_index -= 1
            else:
                new_index -= 1
        pairs.reverse()
        last_matched_current = pairs[-1][1] if pairs else -1
        trailing = [
            index for index in range(last_matched_current + 1, len(current))
        ]
        # Unmatched current rows before/between anchors are OCR drift, not inserts.
        residuals = [
            abs((current[new_i].top - previous[old_i].top) - shift)
            for old_i, new_i in pairs
        ]
        residual = sum(residuals) / len(residuals) if residuals else 0.0
        exact = sum(
            1
            for old_i, new_i in pairs
            if normalized_match_text(previous[old_i].text)
            == normalized_match_text(current[new_i].text)
        )
        confidence = (
            "high"
            if len(pairs) >= 2
            else "low"
            if len(pairs) == 1
            else "none"
        )
        candidate = AlignmentResult(pairs, trailing, shift, confidence, residual)
        if best is None:
            best = candidate
            continue
        key = (
            len(candidate.matched_pairs),
            exact,
            -candidate.residual,
            -(
                candidate.matched_pairs[0][1]
                if candidate.matched_pairs
                else len(current)
            ),
            candidate.matched_pairs[0][0] if candidate.matched_pairs else -1,
        )
        best_exact = sum(
            1
            for old_i, new_i in best.matched_pairs
            if normalized_match_text(previous[old_i].text)
            == normalized_match_text(current[new_i].text)
        )
        best_key = (
            len(best.matched_pairs),
            best_exact,
            -best.residual,
            -(
                best.matched_pairs[0][1]
                if best.matched_pairs
                else len(current)
            ),
            best.matched_pairs[0][0] if best.matched_pairs else -1,
        )
        if key > best_key:
            best = candidate
    assert best is not None
    return best


def visible_overlap(
    previous: Sequence[ChatMessage], current: Sequence[ChatMessage]
) -> int:
    """Compatibility wrapper around alignment for older tests."""

    alignment = align_visible_messages(previous, current)
    if alignment.confidence == "none" and previous:
        return 0
    if not alignment.matched_pairs:
        return 0
    return len(previous) - alignment.matched_pairs[0][0]


def new_visible_messages(
    previous: Sequence[ChatMessage], current: Sequence[ChatMessage]
) -> list[ChatMessage]:
    """Compatibility wrapper: trailing additions only under a confident alignment."""

    alignment = align_visible_messages(previous, current)
    if not previous:
        return list(current)
    if alignment.confidence == "none":
        return []
    return [current[index] for index in alignment.trailing_indices]


def _tracked_messages(tracks: Sequence[TrackedMessage]) -> list[ChatMessage]:
    return [track.message for track in tracks]


def _should_eager_emit(message: ChatMessage) -> bool:
    return (
        message.kind == "player"
        and message.confidence >= EAGER_EMIT_CONFIDENCE
    )


def _sync_visible_track(state: ReconcileState, track: TrackedMessage) -> None:
    for visible in state.visible:
        if visible.track_id == track.track_id:
            visible.emitted = track.emitted
            visible.pending = track.pending
            visible.message = track.message
            visible.sightings = track.sightings
            visible.misses = track.misses
            return


def _emit_track(
    state: ReconcileState, track: TrackedMessage, confirmed: list[ChatMessage]
) -> None:
    """Mark a track emitted and append when it is not a long accidental duplicate."""

    track.pending = False
    if track.emitted:
        _sync_visible_track(state, track)
        return
    norm = normalized_match_text(track.message.text)
    tokens = norm.split()
    # Accidental long-line requeues after OCR drift; short repeats like
    # "gg"/"hi" remain eligible.
    track.emitted = True
    if norm and norm == state.last_emitted_norm and len(tokens) > SHORT_TEXT_LEN:
        _sync_visible_track(state, track)
        return
    state.emitted_count += 1
    state.last_emitted_norm = norm
    confirmed.append(track.message)
    _sync_visible_track(state, track)


def _finalize_or_keep_missed(
    state: ReconcileState, track: TrackedMessage, confirmed: list[ChatMessage]
) -> TrackedMessage | None:
    """Age an unmatched pending track; flush high-confidence player chat on expiry."""

    track.misses += 1
    if track.misses <= PENDING_MISS_LIMIT:
        return track
    if _should_eager_emit(track.message) and not track.emitted:
        _emit_track(state, track, confirmed)
        return None
    state.unconfirmed_drops += 1
    return None


def _age_pending_misses(state: ReconcileState) -> list[ChatMessage]:
    confirmed: list[ChatMessage] = []
    remaining: list[TrackedMessage] = []
    for track in sorted(state.pending, key=lambda item: item.first_seen_order):
        kept = _finalize_or_keep_missed(state, track, confirmed)
        if kept is not None:
            remaining.append(kept)
    state.pending = remaining
    return confirmed


def _stage_tracks(
    state: ReconcileState,
    messages: Sequence[ChatMessage],
    *,
    allow_eager: bool = False,
) -> tuple[list[TrackedMessage], list[ChatMessage]]:
    staged: list[TrackedMessage] = []
    immediate: list[ChatMessage] = []
    for message in messages:
        track = TrackedMessage(
            track_id=state.next_track_id,
            message=message,
            first_seen_order=state.next_track_id,
        )
        state.next_track_id += 1
        staged.append(track)
        if allow_eager and _should_eager_emit(message):
            _emit_track(state, track, immediate)
            # Keep on visible via caller; do not require a second OCR sighting.
            continue
        state.pending.append(track)
    return staged, immediate


def _confirm_pending(
    state: ReconcileState, current: Sequence[ChatMessage]
) -> list[ChatMessage]:
    confirmed: list[ChatMessage] = []
    remaining: list[TrackedMessage] = []
    used: set[int] = set()
    for track in sorted(state.pending, key=lambda item: item.first_seen_order):
        matched_index = None
        best_score = 0.0
        for index, message in enumerate(current):
            if index in used:
                continue
            score = message_match_score(track.message, message)
            if score > best_score:
                best_score = score
                matched_index = index
        if matched_index is None or best_score <= 0:
            kept = _finalize_or_keep_missed(state, track, confirmed)
            if kept is not None:
                remaining.append(kept)
            continue
        used.add(matched_index)
        observation = current[matched_index]
        if observation.confidence >= track.message.confidence:
            track.message = observation
        track.sightings += 1
        track.misses = 0
        ready = track.sightings >= 2 or _should_eager_emit(track.message)
        if not ready:
            remaining.append(track)
            continue
        _emit_track(state, track, confirmed)
    state.pending = remaining
    return confirmed


def _already_tracked(state: ReconcileState, message: ChatMessage) -> bool:
    for track in list(state.visible) + list(state.pending):
        if message_match_score(track.message, message) <= 0:
            continue
        residual = abs(track.message.top - message.top)
        if residual <= max(VERTICAL_RESIDUAL_PX, 0.4 * max(8, track.message.height)):
            return True
    return False


def reconcile_visible(
    state: ReconcileState,
    current: Sequence[ChatMessage],
    *,
    parse_ok: bool = True,
) -> ReconcileResult:
    """Stateful reconciliation with confirmation, resync, and rate guard."""

    if state.started_monotonic <= 0:
        state.started_monotonic = time.monotonic()

    if not parse_ok:
        flushed = _age_pending_misses(state) if state.pending else []
        return ReconcileResult(flushed, state, batch_size=len(flushed), bad_frame=True)

    if not current:
        flushed = _age_pending_misses(state) if state.pending else []
        return ReconcileResult(flushed, state, batch_size=len(flushed), bad_frame=False)

    # Confirm previously staged tracks against this later frame first.
    confirmed = _confirm_pending(state, current) if state.pending else []

    trusted = _tracked_messages(state.visible)
    if not trusted:
        # Baseline: stage the first non-empty window; emit after confirmation
        # (or eager flush if chat fades before a second sighting).
        if not state.pending:
            _stage_tracks(state, current, allow_eager=False)
        state.visible = [
            TrackedMessage(
                track_id=track.track_id,
                message=track.message,
                sightings=track.sightings,
                emitted=track.emitted,
                pending=track.pending,
                first_seen_order=track.first_seen_order,
                misses=track.misses,
            )
            for track in state.pending
        ]
        return ReconcileResult(confirmed, state, batch_size=len(confirmed))

    alignment = align_visible_messages(trusted, current)
    if alignment.confidence == "none":
        state.unanchored_frames += 1
        if state.resync_snapshot is None:
            state.resync_snapshot = list(current)
            return ReconcileResult(confirmed, state, batch_size=len(confirmed))
        second = align_visible_messages(state.resync_snapshot, current)
        if second.confidence != "none" and len(second.matched_pairs) >= max(
            1, len(current) // 2
        ):
            state.resyncs += 1
            state.visible = [
                TrackedMessage(
                    track_id=state.next_track_id + index,
                    message=message,
                    sightings=2,
                    emitted=True,
                    pending=False,
                    first_seen_order=state.next_track_id + index,
                )
                for index, message in enumerate(current)
            ]
            state.next_track_id += len(current)
            state.pending = []
            state.resync_snapshot = None
            return ReconcileResult(confirmed, state, batch_size=len(confirmed))
        state.resync_snapshot = list(current)
        return ReconcileResult(confirmed, state, batch_size=len(confirmed))

    state.unanchored_frames = 0
    state.resync_snapshot = None
    trailing = [
        current[index]
        for index in alignment.trailing_indices
        if not _already_tracked(state, current[index])
    ]
    hold_burst = len(trailing) >= SUSPICIOUS_SUFFIX or (
        len(trailing) > max(1, len(trusted) // 2)
    )
    if trailing:
        if hold_burst:
            state.bulk_suppressed += 1
        staged, immediate = _stage_tracks(state, trailing, allow_eager=True)
        confirmed.extend(immediate)

    refreshed: list[TrackedMessage] = []
    paired_old = {old_i: new_i for old_i, new_i in alignment.matched_pairs}
    for old_index, track in enumerate(state.visible):
        if old_index in paired_old:
            observation = current[paired_old[old_index]]
            if observation.confidence >= track.message.confidence:
                track.message = observation
            track.sightings += 1
            refreshed.append(track)
    pending_ids = {track.track_id for track in state.pending}
    for track in state.pending:
        refreshed.append(
            TrackedMessage(
                track_id=track.track_id,
                message=track.message,
                sightings=track.sightings,
                emitted=track.emitted,
                pending=track.pending,
                first_seen_order=track.first_seen_order,
                misses=track.misses,
            )
        )
    # Eager trailing tracks are not in pending; append them so alignment can
    # treat them as trusted on the next frame.
    if trailing:
        for track in staged:
            if track.track_id in pending_ids:
                continue
            if track.emitted and not track.pending:
                refreshed.append(track)
    state.visible = refreshed or state.visible

    elapsed_min = max((time.monotonic() - state.started_monotonic) / 60.0, 1 / 60)
    rate = state.emitted_count / elapsed_min
    if rate > RATE_LIMIT_PER_MINUTE and state.emitted_count >= 20:
        state.failed = True
        state.failure_reason = (
            f"collector_pipeline_failure: queued rate {rate:.1f}/min "
            f"exceeded {RATE_LIMIT_PER_MINUTE}/min"
        )

    return ReconcileResult(confirmed, state, batch_size=len(confirmed))


def confirm_unchanged_digest(state: ReconcileState) -> list[ChatMessage]:
    """Identical cleaned frames can confirm pending tracks without re-OCR."""

    if not state.pending:
        return []
    current = _tracked_messages(state.visible) or [track.message for track in state.pending]
    return _confirm_pending(state, current)


def image_digest(image, cv2) -> str:
    # Hash the OCR page, not the translucent game pixels. Moving terrain behind
    # unchanged chat should not trigger another expensive Tesseract pass.
    processed = preprocess_for_ocr(image, cv2)
    small = cv2.resize(processed, (96, 48), interpolation=cv2.INTER_AREA)
    fingerprint = (small < 224).astype("uint8")
    return hashlib.sha1(fingerprint.tobytes()).hexdigest()


def live_client_data(path: str, timeout: float = 0.75) -> dict[str, Any] | list[Any] | None:
    """Read Riot's local, read-only Live Client Data API."""

    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(
            f"{LIVE_API}/{path}", timeout=timeout, context=context
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def roster_names(all_game_data: dict[str, Any] | None) -> list[str]:
    """Extract identifiers for in-memory redaction; callers must not persist them."""

    if not all_game_data:
        return []
    names: set[str] = set()
    for player in all_game_data.get("allPlayers", []):
        for key in ("riotId", "riotIdGameName", "summonerName"):
            value = str(player.get(key, "")).strip()
            if value:
                names.add(value)
                if "#" in value:
                    names.add(value.split("#", 1)[0])
    return sorted(names, key=len, reverse=True)


def anonymize_text(text: str, names: Iterable[str] = ()) -> str:
    """Remove known roster names, links, and Riot-ID-shaped identifiers."""

    result = URL_RE.sub("[LINK]", str(text))
    result = RIOT_ID_RE.sub("[PLAYER]", result)
    for name in sorted({name.strip() for name in names if name.strip()}, key=len, reverse=True):
        result = re.sub(
            rf"(?<!\w){re.escape(name)}(?!\w)",
            "[PLAYER]",
            result,
            flags=re.IGNORECASE,
        )
    return " ".join(result.split()).strip()


def session_id() -> str:
    return f"M-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:8]}"


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def command_list_windows(_args: argparse.Namespace) -> None:
    require_windows()
    enable_dpi_awareness()
    _cv2, _mss, _np, _pytesseract, win32gui = import_capture_dependencies()
    print("Visible windows (hwnd, rank, title, x, y, w, h):")
    for hwnd, title, (x, y, width, height) in list_capture_windows(win32gui):
        rank = league_window_rank(title)
        marker = f"rank={rank}" if rank is not None else "      "
        print(f"  {hwnd:>10}  {marker}  {title!r}  ({x}, {y}, {width}, {height})")
    try:
        selected_hwnd, selected_title = find_league_window(win32gui)
    except RuntimeError as exc:
        print(f"\nAuto-select: {exc}")
        return
    print(f"\nAuto-select: hwnd={selected_hwnd} title={selected_title!r}")


def command_calibrate(args: argparse.Namespace) -> None:
    require_windows()
    enable_dpi_awareness()
    cv2, mss, np, pytesseract, win32gui = import_capture_dependencies()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    ensure_tesseract_available(pytesseract)
    if live_client_data("gamestats") is None:
        raise RuntimeError(
            "No active match detected. Finish loading into Practice Tool or a "
            "custom match, open the in-game chat, then calibrate. Lobby, champion "
            "select, and direct-message chat are not valid final-test sources."
        )

    hwnd, title = find_league_window(win32gui, hwnd=args.hwnd)
    rect = client_rect(win32gui, hwnd)
    print(
        f"Capturing window {title!r} (hwnd={hwnd}) "
        f"client_rect=({rect[0]}, {rect[1]}, {rect[2]}, {rect[3]})"
    )
    with open_screen_capture(mss) as sct:
        image = grab_region(sct, np, rect)
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    print(
        "An OpenCV window will open with a screenshot of that League window "
        "(not Cursor). Drag tightly around only the chat text lines (exclude "
        "minimap/abilities/empty UI), then press Enter. Press C to cancel."
    )
    roi = cv2.selectROI("Select League chat region", bgr, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise RuntimeError("Calibration cancelled or empty region selected.")

    _, _, client_width, client_height = rect
    calibration = Calibration(
        window_title=title,
        x_ratio=x / client_width,
        y_ratio=y / client_height,
        width_ratio=width / client_width,
        height_ratio=height / client_height,
        created_at=utc_now(),
        active_match_verified=True,
    )
    save_calibration(calibration)

    preview = image[y : y + height, x : x + width]
    processed = preprocess_for_ocr(preview, cv2)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CALIBRATION_ROI_PATH), cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR))
    cv2.imwrite(str(CALIBRATION_OCR_PATH), processed)
    lines = ocr_lines(preview, cv2, pytesseract)
    print(f"Saved calibration to {CALIBRATION_PATH}")
    print(f"Saved ROI preview to {CALIBRATION_ROI_PATH}")
    print(f"Saved OCR input preview to {CALIBRATION_OCR_PATH}")
    print("OCR preview:")
    if not lines:
        print("  (no text found; open chat and recalibrate if this is unexpected)")
    for line in lines:
        print(f"  [{line.confidence:5.1f}] {line.text}")
    mean_conf = (
        sum(line.confidence for line in lines) / len(lines) if lines else 0.0
    )
    if mean_conf < 55:
        print(
            "Tip: mean OCR confidence is low. Open chat (Enter), tighten the ROI "
            "to text only, and raise chat opacity in League settings if available."
        )


def command_collect(args: argparse.Namespace) -> None:
    require_windows()
    enable_dpi_awareness()
    calibration = load_calibration()
    if calibration.active_match_verified is not True:
        raise RuntimeError(
            "This calibration predates active-match verification. Recalibrate "
            "after loading into Practice Tool or a custom match."
        )
    cv2, mss, np, pytesseract, win32gui = import_capture_dependencies()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    ensure_tesseract_available(pytesseract)

    print("Waiting for a League match. Press Ctrl+C to stop.")
    try:
        while live_client_data("gamestats") is None:
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopped before a match was detected.")
        return

    match_id = session_id()
    session_dir = RAW_DIR / match_id
    screenshots_dir = session_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = session_dir / "candidates.jsonl"
    metadata_path = session_dir / "session.json"
    started_at = utc_now()
    health = {
        "player_candidates": 0,
        "uncertain_candidates": 0,
        "system_filtered": 0,
        "noise_filtered": 0,
        "bad_frames": 0,
        "unanchored_frames": 0,
        "resyncs": 0,
        "bulk_suppressed": 0,
        "unconfirmed_drops": 0,
        "max_batch": 0,
    }
    metadata_path.write_text(
        json.dumps(
            {
                "match_id": match_id,
                "collection_started": started_at,
                "collection_finished": None,
                "candidate_count": 0,
                "review_complete": False,
                "discarded": False,
                "health": health,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    roster = set(roster_names(live_client_data("allgamedata")))
    last_roster_refresh = 0.0
    previous_digest = ""
    candidate_count = 0
    unavailable_reads = 0
    state = ReconcileState(visible=[], pending=[], started_monotonic=time.monotonic())
    pending_screenshot: tuple[Any, str, float | None] | None = None
    print(f"Collecting anonymous match {match_id}.")

    def persist_confirmed(
        confirmed: list[ChatMessage],
        *,
        fallback_image: Any,
        fallback_game_time: float | None,
    ) -> None:
        nonlocal candidate_count, pending_screenshot
        if not confirmed:
            return
        if pending_screenshot is not None:
            shot_image, _digest, shot_time = pending_screenshot
        else:
            shot_image = fallback_image
            shot_time = fallback_game_time
        frame_name = f"{candidate_count + 1:04d}_{int(time.time() * 1000)}.png"
        frame_path = screenshots_dir / frame_name
        if not cv2.imwrite(str(frame_path), shot_image):
            raise RuntimeError(f"Could not save screenshot: {frame_path}")
        records = []
        for message in confirmed:
            candidate_count += 1
            health[
                "uncertain_candidates"
                if message.kind == "uncertain"
                else "player_candidates"
            ] += 1
            records.append(
                {
                    "match_id": match_id,
                    "candidate_order": candidate_count,
                    "text": anonymize_text(message.text, roster),
                    "channel": message.channel,
                    "kind": message.kind,
                    "ocr_confidence": round(message.confidence, 2),
                    "bbox": {
                        "left": message.left,
                        "top": message.top,
                        "width": message.width,
                        "height": message.height,
                    },
                    "observed_at": utc_now(),
                    "game_time_seconds": shot_time,
                    "screenshot_ref": str(frame_path.relative_to(FINAL_TEST_DIR)),
                }
            )
        append_jsonl(candidate_path, records)
        health["max_batch"] = max(health["max_batch"], len(records))
        print(
            f"  captured {len(records)} candidate(s); "
            f"{candidate_count} this match"
        )
        if not state.pending:
            pending_screenshot = None

    try:
        with open_screen_capture(mss) as sct:
            while True:
                game_stats = live_client_data("gamestats")
                if game_stats is None:
                    unavailable_reads += 1
                    if unavailable_reads >= args.end_grace:
                        break
                    time.sleep(args.interval)
                    continue
                unavailable_reads = 0
                if time.monotonic() - last_roster_refresh >= 30:
                    roster.update(roster_names(live_client_data("allgamedata")))
                    last_roster_refresh = time.monotonic()

                try:
                    hwnd, title = find_league_window(win32gui, hwnd=args.hwnd)
                except RuntimeError:
                    time.sleep(args.interval)
                    continue
                ensure_calibration_window(calibration, title)
                try:
                    rect = calibrated_rect(
                        calibration, client_rect(win32gui, hwnd)
                    )
                    image = grab_region(sct, np, rect)
                except RuntimeError:
                    time.sleep(args.interval)
                    continue

                digest = image_digest(image, cv2)
                game_time = (
                    game_stats.get("gameTime")
                    if isinstance(game_stats, dict)
                    else None
                )
                if digest == previous_digest:
                    confirmed = confirm_unchanged_digest(state)
                    persist_confirmed(
                        confirmed,
                        fallback_image=image,
                        fallback_game_time=game_time,
                    )
                    time.sleep(args.interval)
                    continue
                previous_digest = digest

                lines = ocr_lines(image, cv2, pytesseract)
                current_visible, parse_stats = extract_chat_messages(
                    lines, roi_width=image.shape[1]
                )
                health["system_filtered"] += parse_stats.system
                health["noise_filtered"] += parse_stats.noise
                parse_ok = bool(current_visible) or (
                    parse_stats.system + parse_stats.noise > 0
                    and parse_stats.player + parse_stats.uncertain == 0
                )
                # Empty OCR with no filtered rows is a bad/hidden frame.
                if not current_visible and parse_stats.system == 0 and parse_stats.noise == 0:
                    health["bad_frames"] += 1
                    result = reconcile_visible(state, [], parse_ok=False)
                else:
                    result = reconcile_visible(
                        state, current_visible, parse_ok=True
                    )
                state = result.state
                health["unanchored_frames"] = state.unanchored_frames
                health["resyncs"] = state.resyncs
                health["bulk_suppressed"] = state.bulk_suppressed
                health["unconfirmed_drops"] = state.unconfirmed_drops
                # Keep the last frame that still showed chat text so pending
                # lines flushed after fade-out retain a review screenshot.
                if current_visible or state.pending:
                    if current_visible:
                        pending_screenshot = (image.copy(), digest, game_time)

                persist_confirmed(
                    result.confirmed,
                    fallback_image=image,
                    fallback_game_time=game_time,
                )

                if state.failed:
                    print(f"Collector circuit breaker: {state.failure_reason}")
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nCollection stopped by user.")
    finally:
        # Match ended or interrupted: flush high-confidence pending rather than
        # losing chat that never got a second OCR sighting.
        if state.pending and not state.failed and pending_screenshot is not None:
            final_flush: list[ChatMessage] = []
            remaining_pending: list[TrackedMessage] = []
            for track in sorted(
                state.pending, key=lambda item: item.first_seen_order
            ):
                if _should_eager_emit(track.message) and not track.emitted:
                    _emit_track(state, track, final_flush)
                else:
                    remaining_pending.append(track)
            state.pending = remaining_pending
            persist_confirmed(
                final_flush,
                fallback_image=pending_screenshot[0],
                fallback_game_time=pending_screenshot[2],
            )
        elif state.pending and not state.failed:
            state.unconfirmed_drops += len(state.pending)
            state.pending = []
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["collection_finished"] = utc_now()
        metadata["candidate_count"] = candidate_count
        metadata["health"] = health
        if state.failed:
            metadata["discarded"] = True
            metadata["discard_reason"] = state.failure_reason
            metadata["discarded_at"] = utc_now()
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {candidate_count} candidates in {session_dir}")
        if state.failed:
            print("Session auto-marked discarded due to collector failure.")
        else:
            print("Run `python scripts/collect_chat.py review` before assigning labels.")


def session_directories(raw_dir: Path = RAW_DIR) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        (
            path
            for path in raw_dir.iterdir()
            if path.is_dir() and (path / "session.json").exists()
        ),
        key=lambda path: path.name,
    )


def load_session_metadata(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "session.json").read_text(encoding="utf-8"))


def is_discarded(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("discarded"))


def choose_session(requested: str | None) -> Path:
    sessions = session_directories()
    if requested:
        path = RAW_DIR / requested
        if path not in sessions:
            raise RuntimeError(f"Unknown session: {requested}")
        metadata = load_session_metadata(path)
        if is_discarded(metadata):
            raise RuntimeError(
                f"Session {requested} is discarded "
                f"({metadata.get('discard_reason', 'no reason')})."
            )
        return path
    incomplete = []
    for path in sessions:
        metadata = load_session_metadata(path)
        if is_discarded(metadata):
            continue
        if not metadata.get("review_complete"):
            incomplete.append(path)
    if not incomplete:
        raise RuntimeError("No unreviewed sessions found.")
    return incomplete[0]


def corrected_text_is_safe(text: str) -> bool:
    if URL_RE.search(text) or RIOT_ID_RE.search(text):
        return False
    if PLAYER_PREFIX_SHAPE_RE.search(text):
        return False
    return True


DROP_REASON_ALIASES = {
    "s": "system",
    "system": "system",
    "o": "ocr duplicate",
    "ocr": "ocr duplicate",
    "ocr duplicate": "ocr duplicate",
    "e": "empty",
    "empty": "empty",
    "n": "non-player",
    "non-player": "non-player",
    "nonplayer": "non-player",
}
# Remembered across candidates in one review session so Enter repeats the
# last drop reason (defaults to system, the most common exclusion).
_last_drop_reason = "system"


def resolve_drop_reason(raw: str) -> str | None:
    """Map a short review answer to a protocol exclusion reason."""

    return DROP_REASON_ALIASES.get(raw.strip().lower())


def review_candidate(
    record: dict[str, Any], input_fn=input
) -> tuple[str, str, str]:
    """Return decision, corrected text, and objective exclusion reason."""

    global _last_drop_reason
    print(
        f"\n#{record['candidate_order']} [{record['kind']}; "
        f"OCR {record['ocr_confidence']:.1f}]"
    )
    print(record["text"])
    print(f"Source: {FINAL_TEST_DIR / record['screenshot_ref']}")
    require_explicit = record["kind"] != "player"
    prompt = (
        "[k]eep, [e]dit+keep, [d]rop, "
        "[s]ystem-drop, [o]cr-dup, [n]on-player, [q]uit"
    )
    if require_explicit:
        prompt += " (no default for uncertain/system)"
    else:
        prompt += " (default k)"
    while True:
        answer = input_fn(f"{prompt}: ").strip().lower()
        if not answer:
            if require_explicit:
                print("Choose explicitly; uncertain/system lines have no default keep.")
                continue
            answer = "k"
        if answer == "k":
            corrected = anonymize_text(record["text"])
            if not corrected:
                print("Anonymization left an empty message; drop or edit it.")
                continue
            if not corrected_text_is_safe(corrected):
                print(
                    "Kept text still looks like a player prefix, URL, or Riot ID. "
                    "Edit to the message body only."
                )
                continue
            return "keep", corrected, ""
        if answer == "e":
            corrected = input_fn(
                "Corrected message body only (no speaker/prefix): "
            ).strip()
            corrected = anonymize_text(corrected)
            if not corrected:
                print("Corrected text cannot be empty.")
                continue
            if not corrected_text_is_safe(corrected):
                print("Rejecting prefix-shaped, URL, or Riot-ID text.")
                continue
            return "keep", corrected, ""
        if answer in {"s", "o", "n"}:
            reason = resolve_drop_reason(answer)
            assert reason is not None
            _last_drop_reason = reason
            return "drop", "", reason
        if answer == "d":
            reason_raw = input_fn(
                f"Reason [s]ystem/[o]CR dup/[e]mpty/[n]on-player "
                f"(Enter={_last_drop_reason}): "
            ).strip().lower()
            if not reason_raw:
                reason_raw = _last_drop_reason
            reason = resolve_drop_reason(reason_raw)
            if reason is None:
                print("Use s/o/e/n or system/ocr duplicate/empty/non-player.")
                continue
            _last_drop_reason = reason
            return "drop", "", reason
        if answer == "q":
            return "quit", "", ""
        print("Unknown choice; use k/e/d/s/o/n/q.")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_reviewed(
    raw_dir: Path = RAW_DIR,
    export_path: Path = EXPORT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    audit_path: Path = AUDIT_PATH,
) -> tuple[int, int]:
    """Rebuild the sanitized final CSV and audit from reviewed sessions."""

    kept: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []

    for directory in session_directories(raw_dir):
        metadata = load_session_metadata(directory)
        if is_discarded(metadata):
            discarded.append(
                {
                    "match_id": metadata["match_id"],
                    "reason": metadata.get(
                        "discard_reason", "collector_pipeline_failure"
                    ),
                    "discarded_at": metadata.get("discarded_at"),
                }
            )
            continue
        reviewed = read_jsonl(directory / "reviewed.jsonl")
        if not metadata.get("review_complete"):
            continue
        accepted = 0
        for record in reviewed:
            if record.get("decision") == "keep":
                text = anonymize_text(record["corrected_text"])
                if not corrected_text_is_safe(text):
                    raise RuntimeError(
                        f"Export blocked: unsafe corrected text in "
                        f"{metadata['match_id']}#{record['candidate_order']}"
                    )
                accepted += 1
                kept.append(
                    {
                        "match_id": metadata["match_id"],
                        "candidate_order": int(record["candidate_order"]),
                        "text": text,
                    }
                )
            elif record.get("decision") == "drop":
                exclusions.append(
                    {
                        "match_id": metadata["match_id"],
                        "candidate_order": int(record["candidate_order"]),
                        "reason": record["exclusion_reason"],
                    }
                )
        manifests.append(
            {
                "match_id": metadata["match_id"],
                "collection_started": metadata["collection_started"],
                "collection_finished": metadata["collection_finished"],
                "candidate_count": metadata["candidate_count"],
                "eligible_count": accepted,
            }
        )

    kept.sort(key=lambda row: (row["match_id"], row["candidate_order"]))
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["match_id", "message_order", "text", "label", "notes"]
        )
        writer.writeheader()
        per_match_order: dict[str, int] = {}
        for row in kept:
            match_id = row["match_id"]
            per_match_order[match_id] = per_match_order.get(match_id, 0) + 1
            writer.writerow(
                {
                    "match_id": match_id,
                    "message_order": per_match_order[match_id],
                    "text": row["text"],
                    "label": "",
                    "notes": "",
                }
            )

    manifest = {
        "generated_at": utc_now(),
        "reviewed_matches": len(manifests),
        "eligible_messages": len(kept),
        "discarded_matches": len(discarded),
        "targets": {
            "matches_minimum": TARGET_MATCHES,
            "messages_minimum": TARGET_MIN_MESSAGES,
            "messages_maximum": TARGET_MAX_MESSAGES,
        },
        "sessions": manifests,
        "discarded_sessions": discarded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now(),
                "excluded_count": len(exclusions),
                "exclusions": exclusions,
                "discarded_sessions": discarded,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(manifests), len(kept)


def print_progress(
    matches: int, messages: int, discarded: int = 0
) -> None:
    print(f"Final-test progress: {messages} eligible messages from {matches} matches.")
    if discarded:
        print(f"  Discarded technical-failure sessions: {discarded}")
    if matches < TARGET_MATCHES:
        print(f"  Need at least {TARGET_MATCHES - matches} more reviewed matches.")
    if messages < TARGET_MIN_MESSAGES:
        print(f"  Need at least {TARGET_MIN_MESSAGES - messages} more eligible messages.")
    elif messages > TARGET_MAX_MESSAGES:
        print(
            "  Message target exceeded. Do not cherry-pick removals; document the "
            "consecutive stopping rule used."
        )
    elif matches >= TARGET_MATCHES:
        print("  Protocol collection targets are satisfied.")


def diagnose_session(
    directory: Path, cv2, pytesseract
) -> dict[str, Any]:
    """Replay unique screenshots through the current pipeline without writing."""

    candidates = read_jsonl(directory / "candidates.jsonl")
    ordered_refs: list[str] = []
    seen_refs: set[str] = set()
    for record in candidates:
        ref = str(record.get("screenshot_ref", ""))
        if ref and ref not in seen_refs:
            seen_refs.add(ref)
            ordered_refs.append(ref)
    if not ordered_refs:
        shot_dir = directory / "screenshots"
        ordered_refs = [
            str(path.relative_to(FINAL_TEST_DIR)).replace("\\", "/")
            for path in sorted(shot_dir.glob("*.png"))
        ]

    state = ReconcileState(visible=[], pending=[], started_monotonic=time.monotonic())
    health = {
        "frames": 0,
        "player_candidates": 0,
        "uncertain_candidates": 0,
        "system_filtered": 0,
        "noise_filtered": 0,
        "bad_frames": 0,
        "emitted": 0,
        "max_batch": 0,
        "unanchored_frames": 0,
        "resyncs": 0,
        "bulk_suppressed": 0,
        "adjacent_duplicates": 0,
    }
    emitted_texts: list[str] = []
    rejected_examples: list[str] = []
    previous_digest = ""

    for ref in ordered_refs:
        path = FINAL_TEST_DIR / ref
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        health["frames"] += 1
        digest = image_digest(image, cv2)
        if digest == previous_digest:
            confirmed = confirm_unchanged_digest(state)
            if confirmed:
                health["emitted"] += len(confirmed)
                health["max_batch"] = max(health["max_batch"], len(confirmed))
                for message in confirmed:
                    key = normalized_match_text(message.text)
                    tokens = key.split()
                    if (
                        emitted_texts
                        and key == emitted_texts[-1]
                        and len(tokens) > SHORT_TEXT_LEN
                    ):
                        health["adjacent_duplicates"] += 1
                    emitted_texts.append(key)
                    health[
                        "uncertain_candidates"
                        if message.kind == "uncertain"
                        else "player_candidates"
                    ] += 1
            continue
        previous_digest = digest
        lines = ocr_lines(image, cv2, pytesseract)
        current_visible, parse_stats = extract_chat_messages(
            lines, roi_width=image.shape[1]
        )
        health["system_filtered"] += parse_stats.system
        health["noise_filtered"] += parse_stats.noise
        if not current_visible and parse_stats.system == 0 and parse_stats.noise == 0:
            health["bad_frames"] += 1
            result = reconcile_visible(state, [], parse_ok=False)
        else:
            result = reconcile_visible(state, current_visible, parse_ok=True)
        state = result.state
        health["unanchored_frames"] = state.unanchored_frames
        health["resyncs"] = state.resyncs
        health["bulk_suppressed"] = state.bulk_suppressed
        if not current_visible and parse_stats.noise:
            if len(rejected_examples) < 5:
                rejected_examples.append(ref)
        if result.confirmed:
            health["emitted"] += len(result.confirmed)
            health["max_batch"] = max(health["max_batch"], len(result.confirmed))
            for message in result.confirmed:
                key = normalized_match_text(message.text)
                tokens = key.split()
                if (
                    emitted_texts
                    and key == emitted_texts[-1]
                    and len(tokens) > SHORT_TEXT_LEN
                ):
                    health["adjacent_duplicates"] += 1
                emitted_texts.append(key)
                health[
                    "uncertain_candidates"
                    if message.kind == "uncertain"
                    else "player_candidates"
                ] += 1

    duration_min = max(health["frames"] * 0.75 / 60.0, 1 / 60)
    health["candidate_rate_per_min"] = round(health["emitted"] / duration_min, 2)
    health["adjacent_duplicate_rate"] = round(
        health["adjacent_duplicates"] / max(1, health["emitted"]), 4
    )
    health["rejected_examples"] = rejected_examples
    health["fingerprint"] = hashlib.sha1(
        json.dumps(
            {
                "emitted": health["emitted"],
                "texts": emitted_texts,
                "system_filtered": health["system_filtered"],
                "noise_filtered": health["noise_filtered"],
                "max_batch": health["max_batch"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return health


def command_diagnose(args: argparse.Namespace) -> None:
    directory = RAW_DIR / args.session
    if not directory.exists():
        raise RuntimeError(f"Unknown session: {args.session}")
    cv2, _mss, _np, pytesseract, _win32gui = import_capture_dependencies()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    ensure_tesseract_available(pytesseract)
    first = diagnose_session(directory, cv2, pytesseract)
    second = diagnose_session(directory, cv2, pytesseract)
    print(json.dumps(first, indent=2))
    if first["fingerprint"] != second["fingerprint"]:
        raise RuntimeError("Diagnose replay was not deterministic across two runs.")
    print("Replay fingerprint matched on second run.")
    if first["emitted"] > 150:
        print("WARNING: emitted candidates exceed 150 acceptance threshold.")
    if first["max_batch"] > 4:
        print("WARNING: max batch exceeds 4.")
    if first["adjacent_duplicate_rate"] > 0.02:
        print("WARNING: adjacent duplicate rate exceeds 2%.")


def command_discard_session(args: argparse.Namespace) -> None:
    directory = RAW_DIR / args.session
    if not directory.exists():
        raise RuntimeError(f"Unknown session: {args.session}")
    metadata_path = directory / "session.json"
    metadata = load_session_metadata(directory)
    metadata["discarded"] = True
    metadata["discard_reason"] = args.reason
    metadata["discarded_at"] = utc_now()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Marked {args.session} discarded ({args.reason}).")
    print("Screenshots remain available for diagnose until deleted.")


def command_review(args: argparse.Namespace) -> None:
    directory = choose_session(args.session)
    candidates = read_jsonl(directory / "candidates.jsonl")
    reviewed_path = directory / "reviewed.jsonl"
    reviewed = read_jsonl(reviewed_path)
    start = len(reviewed)
    print(f"Reviewing {directory.name}: {len(candidates)} candidates, {start} complete.")

    for record in candidates[start:]:
        decision, corrected_text, reason = review_candidate(record)
        if decision == "quit":
            print("Review paused; run the command again to resume.")
            return
        reviewed_record = {
            **record,
            "decision": decision,
            "corrected_text": corrected_text,
            "exclusion_reason": reason,
            "reviewed_at": utc_now(),
        }
        append_jsonl(reviewed_path, [reviewed_record])
        reviewed.append(reviewed_record)

    metadata_path = directory / "session.json"
    metadata = load_session_metadata(directory)
    metadata["review_complete"] = True
    metadata["review_finished"] = utc_now()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    matches, messages = export_reviewed()
    print(f"Sanitized export rebuilt at {EXPORT_PATH}")
    discarded = 0
    if MANIFEST_PATH.exists():
        discarded = int(
            json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get(
                "discarded_matches", 0
            )
        )
    print_progress(matches, messages, discarded=discarded)

    if args.delete_raw:
        delete = "y"
    else:
        delete = input(
            "Delete raw screenshots now that sanitized review is complete? [y/N]: "
        ).strip().lower()
    if delete == "y":
        shutil.rmtree(directory / "screenshots", ignore_errors=True)
        print("Raw screenshots deleted. Reviewed text and audit metadata were retained.")


def command_status(_args: argparse.Namespace) -> None:
    discarded = sum(
        1
        for path in session_directories()
        if is_discarded(load_session_metadata(path))
    )
    if not MANIFEST_PATH.exists():
        print_progress(0, 0, discarded=discarded)
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print_progress(
        int(manifest.get("reviewed_matches", 0)),
        int(manifest.get("eligible_messages", 0)),
        discarded=int(manifest.get("discarded_matches", discarded)),
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Passively collect visible LoL chat for human final-test labeling."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="select the chat region")
    calibrate.add_argument("--tesseract-cmd", help="path to tesseract.exe")
    calibrate.add_argument(
        "--hwnd",
        type=int,
        default=None,
        help="optional Win32 window handle from list-windows",
    )
    calibrate.set_defaults(func=command_calibrate)

    collect = subparsers.add_parser("collect", help="collect one match")
    collect.add_argument("--interval", type=positive_float, default=0.75)
    collect.add_argument(
        "--end-grace",
        type=positive_int,
        default=5,
        help="consecutive unavailable API reads before ending the session",
    )
    collect.add_argument("--tesseract-cmd", help="path to tesseract.exe")
    collect.add_argument(
        "--hwnd",
        type=int,
        default=None,
        help="optional Win32 window handle from list-windows",
    )
    collect.set_defaults(func=command_collect)

    list_windows = subparsers.add_parser(
        "list-windows",
        help="list visible windows and the auto-selected League hwnd",
    )
    list_windows.set_defaults(func=command_list_windows)

    diagnose = subparsers.add_parser(
        "diagnose",
        help="replay a session's screenshots through the current pipeline",
    )
    diagnose.add_argument("--session", required=True, help="anonymous match ID")
    diagnose.add_argument("--tesseract-cmd", help="path to tesseract.exe")
    diagnose.set_defaults(func=command_diagnose)

    discard = subparsers.add_parser(
        "discard-session",
        help="mark a session as a technical failure excluded from review/export",
    )
    discard.add_argument("--session", required=True, help="anonymous match ID")
    discard.add_argument(
        "--reason",
        default="collector_pipeline_failure",
        help="objective discard reason",
    )
    discard.set_defaults(func=command_discard_session)

    review = subparsers.add_parser("review", help="correct and export OCR candidates")
    review.add_argument("--session", help="anonymous match ID; defaults to first pending")
    review.add_argument(
        "--delete-raw",
        action="store_true",
        help="delete screenshots after completed review without prompting",
    )
    review.set_defaults(func=command_review)

    status = subparsers.add_parser("status", help="show protocol collection counters")
    status.set_defaults(func=command_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit("\nStopped by user.") from None
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()

