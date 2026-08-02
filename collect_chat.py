"""Passive screen-OCR collector for the fresh League of Legends final test.

The Riot APIs do not expose in-game team/all chat. This tool therefore reads
only pixels already visible to the user. It never injects input, reads process
memory, or assigns toxicity labels.

Typical use:

    python collect_chat.py calibrate
    python collect_chat.py collect
    python collect_chat.py review
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
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

ROOT = Path(__file__).resolve().parent
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
TARGET_MIN_MESSAGES = 200
TARGET_MAX_MESSAGES = 300
TARGET_MATCHES = 20
CALIBRATION_ROI_PATH = RAW_DIR / "calibration_roi.png"
CALIBRATION_OCR_PATH = RAW_DIR / "calibration_ocr_input.png"

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
RIOT_ID_RE = re.compile(
    r"(?<!\w)[A-Za-z0-9][A-Za-z0-9._'-]{1,22}#[A-Za-z0-9]{3,6}(?!\w)"
)
TIMESTAMP_RE = r"(?:\[\s*\d{1,2}:\d{2}\s*\]\s*)?"
CHAT_PREFIX_RE = re.compile(
    rf"^\s*{TIMESTAMP_RE}"
    r"(?:\[\s*(?P<channel>all|team|party)\s*\]\s*)?"
    r"(?P<speaker>[^:\r\n]{1,64}?)"
    r"(?:\s+\([^)]+\))?\s*:\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)
SYSTEM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bhas (?:slain|destroyed|joined|left|disconnected|reconnected)\b",
        r"\bis on a killing spree\b",
        r"\b(?:shutdown|first blood|ace!)\b",
        r"\b(?:enemy|ally) (?:slain|missing)\b",
        r"\bpurchased\b",
        r"\b(?:turret|inhibitor) (?:destroyed|respawned)\b",
    )
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
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            raise RuntimeError(f"Window handle {hwnd} is not a visible window.")
        title = win32gui.GetWindowText(hwnd).strip() or f"hwnd:{hwnd}"
        return hwnd, title

    # (rank, area, hwnd, title)
    candidates: list[tuple[int, int, int, str]] = []

    def visit(candidate_hwnd: int, _extra: object) -> None:
        if not win32gui.IsWindowVisible(candidate_hwnd):
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
            "Use `python collect_chat.py list-windows` to inspect titles."
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
        raise RuntimeError("No calibration found. Run `python collect_chat.py calibrate`.")
    return Calibration(**json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")))


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


def ocr_lines(image, cv2, pytesseract) -> list[OCRLine]:
    """Extract visual text lines and confidence/bounding-box provenance."""

    processed = preprocess_for_ocr(image, cv2)
    data = pytesseract.image_to_data(
        processed,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )
    grouped: dict[tuple[int, int, int], list[tuple[str, float, int, int, int, int]]] = {}
    for index, word in enumerate(data["text"]):
        word = str(word).strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if not word or confidence < MIN_OCR_CONFIDENCE:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped.setdefault(key, []).append(
            (
                word,
                confidence,
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
            )
        )

    lines: list[OCRLine] = []
    for words in grouped.values():
        words.sort(key=lambda item: item[2])
        left = min(item[2] for item in words) - OCR_BORDER
        top = min(item[3] for item in words) - OCR_BORDER
        right = max(item[2] + item[4] for item in words) - OCR_BORDER
        bottom = max(item[3] + item[5] for item in words) - OCR_BORDER
        lines.append(
            OCRLine(
                text=" ".join(item[0] for item in words),
                confidence=sum(item[1] for item in words) / len(words),
                left=round(left / OCR_SCALE),
                top=round(top / OCR_SCALE),
                width=round((right - left) / OCR_SCALE),
                height=round((bottom - top) / OCR_SCALE),
            )
        )
    return sorted(lines, key=lambda line: (line.top, line.left))


def looks_system_generated(text: str) -> bool:
    return any(pattern.search(text) for pattern in SYSTEM_PATTERNS)


def extract_chat_messages(lines: Sequence[OCRLine]) -> list[ChatMessage]:
    """Parse player prefixes and join visual continuation lines.

    Unrecognized lines remain in the review queue as ``uncertain`` rather than
    being silently discarded.
    """

    messages: list[ChatMessage] = []
    for line in lines:
        match = CHAT_PREFIX_RE.match(line.text)
        if match:
            text = match.group("text").strip()
            kind = "system" if looks_system_generated(text) else "player"
            messages.append(
                ChatMessage(
                    text=text,
                    channel=(match.group("channel") or "unknown").lower(),
                    confidence=line.confidence,
                    kind=kind,
                    left=line.left,
                    top=line.top,
                    width=line.width,
                    height=line.height,
                )
            )
            continue

        if messages and not looks_system_generated(line.text):
            previous = messages[-1]
            messages[-1] = ChatMessage(
                text=f"{previous.text} {line.text.strip()}".strip(),
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
        else:
            messages.append(
                ChatMessage(
                    text=line.text.strip(),
                    channel="unknown",
                    confidence=line.confidence,
                    kind="system" if looks_system_generated(line.text) else "uncertain",
                    left=line.left,
                    top=line.top,
                    width=line.width,
                    height=line.height,
                )
            )
    return [message for message in messages if message.text]


def normalized_match_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def messages_match(previous: ChatMessage, current: ChatMessage) -> bool:
    first = normalized_match_text(previous.text)
    second = normalized_match_text(current.text)
    if not first or not second:
        return False
    return difflib.SequenceMatcher(None, first, second).ratio() >= 0.82


def visible_overlap(
    previous: Sequence[ChatMessage], current: Sequence[ChatMessage]
) -> int:
    """Find prior-suffix/current-prefix overlap using text and line geometry."""

    for size in range(min(len(previous), len(current)), 0, -1):
        old = previous[-size:]
        new = current[:size]
        if not all(messages_match(a, b) for a, b in zip(old, new)):
            continue
        offsets = [b.top - a.top for a, b in zip(old, new)]
        typical_height = max(8, round(sum(a.height for a in old) / len(old)))
        if max(offsets) - min(offsets) <= typical_height * 2:
            return size
    return 0


def new_visible_messages(
    previous: Sequence[ChatMessage], current: Sequence[ChatMessage]
) -> list[ChatMessage]:
    return list(current[visible_overlap(previous, current) :])


def image_digest(image, cv2) -> str:
    small = cv2.resize(image, (48, 24), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes()).hexdigest()


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
    cv2, mss, np, pytesseract, win32gui = import_capture_dependencies()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    ensure_tesseract_available(pytesseract)

    print("Waiting for a League match. Press Ctrl+C to stop.")
    while live_client_data("gamestats") is None:
        time.sleep(2)

    match_id = session_id()
    session_dir = RAW_DIR / match_id
    screenshots_dir = session_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = session_dir / "candidates.jsonl"
    metadata_path = session_dir / "session.json"
    started_at = utc_now()
    metadata_path.write_text(
        json.dumps(
            {
                "match_id": match_id,
                "collection_started": started_at,
                "collection_finished": None,
                "candidate_count": 0,
                "review_complete": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    roster = set(roster_names(live_client_data("allgamedata")))
    last_roster_refresh = 0.0
    previous_visible: list[ChatMessage] = []
    previous_digest = ""
    candidate_count = 0
    unavailable_reads = 0
    print(f"Collecting anonymous match {match_id}.")

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
                    hwnd, _title = find_league_window(win32gui, hwnd=args.hwnd)
                    rect = calibrated_rect(
                        calibration, client_rect(win32gui, hwnd)
                    )
                    image = grab_region(sct, np, rect)
                except RuntimeError:
                    time.sleep(args.interval)
                    continue

                digest = image_digest(image, cv2)
                if digest == previous_digest:
                    time.sleep(args.interval)
                    continue
                previous_digest = digest

                current_visible = extract_chat_messages(
                    ocr_lines(image, cv2, pytesseract)
                )
                added = new_visible_messages(previous_visible, current_visible)
                if current_visible:
                    previous_visible = current_visible
                if not added:
                    time.sleep(args.interval)
                    continue

                frame_name = (
                    f"{candidate_count + 1:04d}_{int(time.time() * 1000)}.png"
                )
                frame_path = screenshots_dir / frame_name
                cv2.imwrite(str(frame_path), image)
                game_time = (
                    game_stats.get("gameTime")
                    if isinstance(game_stats, dict)
                    else None
                )

                records = []
                for message in added:
                    candidate_count += 1
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
                            "game_time_seconds": game_time,
                            "screenshot_ref": str(
                                frame_path.relative_to(FINAL_TEST_DIR)
                            ),
                        }
                    )
                append_jsonl(candidate_path, records)
                print(
                    f"  captured {len(records)} candidate(s); "
                    f"{candidate_count} this match"
                )
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nCollection stopped by user.")
    finally:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["collection_finished"] = utc_now()
        metadata["candidate_count"] = candidate_count
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {candidate_count} candidates in {session_dir}")
        print("Run `python collect_chat.py review` before assigning labels.")


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


def choose_session(requested: str | None) -> Path:
    sessions = session_directories()
    if requested:
        path = RAW_DIR / requested
        if path not in sessions:
            raise RuntimeError(f"Unknown session: {requested}")
        return path
    incomplete = []
    for path in sessions:
        metadata = json.loads((path / "session.json").read_text(encoding="utf-8"))
        if not metadata.get("review_complete"):
            incomplete.append(path)
    if not incomplete:
        raise RuntimeError("No unreviewed sessions found.")
    return incomplete[0]


def review_candidate(
    record: dict[str, Any], input_fn=input
) -> tuple[str, str, str]:
    """Return decision, corrected text, and objective exclusion reason."""

    print(
        f"\n#{record['candidate_order']} [{record['kind']}; "
        f"OCR {record['ocr_confidence']:.1f}]"
    )
    print(record["text"])
    print(f"Source: {FINAL_TEST_DIR / record['screenshot_ref']}")
    default = "k" if record["kind"] == "player" else "d"
    while True:
        answer = input_fn(
            f"[k]eep, [e]dit+keep, [d]rop, [q]uit (default {default}): "
        ).strip().lower() or default
        if answer == "k":
            corrected = anonymize_text(record["text"])
            if corrected:
                return "keep", corrected, ""
            print("Anonymization left an empty message; drop or edit it.")
        if answer == "e":
            corrected = input_fn("Corrected message text: ").strip()
            corrected = anonymize_text(corrected)
            if corrected:
                return "keep", corrected, ""
            print("Corrected text cannot be empty.")
        elif answer == "d":
            reason = input_fn(
                "Reason [system/empty/non-player/OCR duplicate]: "
            ).strip().lower()
            if reason not in {"system", "empty", "non-player", "ocr duplicate"}:
                print("Use one of the objective protocol exclusions shown.")
                continue
            return "drop", "", reason
        elif answer == "q":
            return "quit", "", ""


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

    for directory in session_directories(raw_dir):
        metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        reviewed = read_jsonl(directory / "reviewed.jsonl")
        if not metadata.get("review_complete"):
            continue
        accepted = 0
        for record in reviewed:
            if record.get("decision") == "keep":
                accepted += 1
                kept.append(
                    {
                        "match_id": metadata["match_id"],
                        "candidate_order": int(record["candidate_order"]),
                        "text": anonymize_text(record["corrected_text"]),
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
        "targets": {
            "matches_minimum": TARGET_MATCHES,
            "messages_minimum": TARGET_MIN_MESSAGES,
            "messages_maximum": TARGET_MAX_MESSAGES,
        },
        "sessions": manifests,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now(),
                "excluded_count": len(exclusions),
                "exclusions": exclusions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(manifests), len(kept)


def print_progress(matches: int, messages: int) -> None:
    print(f"Final-test progress: {messages} eligible messages from {matches} matches.")
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
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["review_complete"] = True
    metadata["review_finished"] = utc_now()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    matches, messages = export_reviewed()
    print(f"Sanitized export rebuilt at {EXPORT_PATH}")
    print_progress(matches, messages)

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
    if not MANIFEST_PATH.exists():
        print_progress(0, 0)
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print_progress(
        int(manifest.get("reviewed_matches", 0)),
        int(manifest.get("eligible_messages", 0)),
    )


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
    collect.add_argument("--interval", type=float, default=0.75)
    collect.add_argument(
        "--end-grace",
        type=int,
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
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
