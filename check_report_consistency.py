"""Verify that report macros and prose contain only canonical metric values."""
from __future__ import annotations

import json
from pathlib import Path
import re

import config


def main() -> None:
    report_dir = config.ROOT / "reports" / "progress"
    manifest_path = config.ARTIFACTS_DIR / "report_metric_manifest.json"
    generated_path = report_dir / "generated_metrics.tex"
    report_path = report_dir / "main.tex"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated = generated_path.read_text(encoding="utf-8")
    report = report_path.read_text(encoding="utf-8")

    errors = []
    for name, expected in manifest.items():
        pattern = re.compile(
            rf"\\newcommand\{{\\{re.escape(name)}\}}\{{(.*?)\}}"
        )
        match = pattern.search(generated)
        if not match or match.group(1) != expected:
            errors.append(
                f"{name}: expected {expected!r}, found "
                f"{None if match is None else match.group(1)!r}"
            )
    if r"\input{generated_metrics.tex}" not in report:
        errors.append("main.tex does not input generated_metrics.tex")

    forbidden_stale_phrases = [
        "select the lowest validation-loss checkpoint",
        "best checkpoint occurred at epoch 3",
        "Packed LSTM & 0.867",
        "the LSTM averaged F1",
        "reserve surrounding-message context for the final project",
        "mean F1 remained",
    ]
    for phrase in forbidden_stale_phrases:
        if phrase in report:
            errors.append(f"stale report phrase remains: {phrase!r}")
    if errors:
        raise AssertionError("\n".join(errors))
    print(f"Verified {len(manifest)} generated report values.")


if __name__ == "__main__":
    main()
