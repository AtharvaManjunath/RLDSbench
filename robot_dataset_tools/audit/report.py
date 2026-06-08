"""Audit report assembly and serialization."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from robot_dataset_tools.audit.schema import validate_episodes
from robot_dataset_tools.audit.statistics import compute_statistics
from robot_dataset_tools.models import Episode
from robot_dataset_tools.utils.arrays import json_safe


def build_audit_report(
    path: str,
    format_name: str,
    episodes: list[Episode],
    strict: bool = False,
    benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues = validate_episodes(episodes, strict=strict)
    total_steps = sum(len(ep.steps) for ep in episodes)
    missing_counts: dict[str, int] = {}
    for issue in issues:
        if issue.code.startswith("MISSING_"):
            missing_counts[issue.code] = missing_counts.get(issue.code, 0) + 1
    stats = compute_statistics(episodes)
    report = {
        "dataset_path": path,
        "detected_format": format_name,
        "episode_count": len(episodes),
        "step_count": total_steps,
        "schema_issues": [issue.to_dict() for issue in issues],
        "missing_field_counts": missing_counts,
        "field_coverage": {
            "required": {
                "action": 1.0 - (missing_counts.get("MISSING_ACTION", 0) / total_steps if total_steps else 0.0),
                "observation": 1.0
                - (missing_counts.get("MISSING_OBSERVATION", 0) / total_steps if total_steps else 0.0),
            },
            "optional": {"language_instruction": stats["language"]["coverage_step_fraction"]},
        },
        "episode_lengths": stats["episode_lengths"],
        "action_stats": stats["actions"],
        "language_instruction_coverage": stats["language"],
        "observation_keys": stats["observations"]["keys"],
        "observation_coverage": stats["observations"]["coverage"],
        "sample_episode_ids": [ep.episode_id for ep in episodes[: min(5, len(episodes))]],
        "warnings": [i.message for i in issues if i.severity == "warning"],
        "errors": [i.message for i in issues if i.severity == "error"],
    }
    if benchmark is not None:
        report["benchmark"] = benchmark
    return json_safe(report)


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(json_safe(report), allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_html_report(report: dict[str, Any], path: str | Path) -> None:
    title = f"Robot Dataset Audit: {report['detected_format']}"
    rows = "\n".join(
        f"<tr><td>{html.escape(issue.get('severity', ''))}</td><td>{html.escape(issue.get('code', ''))}</td>"
        f"<td>{html.escape(issue.get('episode_id', ''))}</td><td>{html.escape(issue.get('message', ''))}</td></tr>"
        for issue in report.get("schema_issues", [])
    )
    cards = {
        "Episodes": report["episode_count"],
        "Steps": report["step_count"],
        "Format": report["detected_format"],
        "Errors": len(report.get("errors", [])),
    }
    card_html = "".join(
        f"<div class='card'><span>{html.escape(k)}</span><strong>{html.escape(str(v))}</strong></div>"
        for k, v in cards.items()
    )
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;color:#1f2937;background:#f8fafc}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:18px 0}}
.card{{background:white;border:1px solid #dbe3ef;border-radius:8px;padding:14px}}
.card span{{display:block;color:#64748b;font-size:13px}} .card strong{{font-size:24px}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #dbe3ef;padding:8px;text-align:left}}th{{background:#eaf0f8}}
pre{{background:white;border:1px solid #dbe3ef;border-radius:8px;padding:12px;overflow:auto}}
</style></head><body>
<h1>{html.escape(title)}</h1><p>{html.escape(report["dataset_path"])}</p>
<div class="cards">{card_html}</div>
<h2>Episode Lengths</h2><pre>{html.escape(json.dumps(report["episode_lengths"], indent=2))}</pre>
<h2>Action Stats</h2><pre>{html.escape(json.dumps(report["action_stats"], indent=2))}</pre>
<h2>Language Coverage</h2><pre>{html.escape(json.dumps(report["language_instruction_coverage"], indent=2))}</pre>
<h2>Issues</h2><table><tr><th>Severity</th><th>Code</th><th>Episode</th><th>Message</th></tr>{rows}</table>
</body></html>"""
    Path(path).write_text(content, encoding="utf-8")
