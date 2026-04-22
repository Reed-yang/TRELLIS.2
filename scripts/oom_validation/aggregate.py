"""Aggregate OOM validation JSONs into a compact markdown summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_float(value, fmt: str = "{:.2f}") -> str:
    if value is None:
        return "NA"
    try:
        return fmt.format(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _fmt_int(value) -> str:
    if value is None:
        return "NA"
    try:
        return f"{int(value):,}"
    except Exception:  # noqa: BLE001
        return str(value)


def _dominant_stage(stage_walls: dict) -> str:
    if not stage_walls:
        return "NA"
    name, wall = max(stage_walls.items(), key=lambda kv: kv[1])
    return f"{name} ({wall:.2f}s)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing worker.py output *.json")
    parser.add_argument("--summary_path", default=None,
                        help="Markdown output path (default: <results_dir>/summary.md)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_path = Path(args.summary_path) if args.summary_path else results_dir / "summary.md"

    payloads: list[dict] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            payloads.append(json.loads(p.read_text()))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to parse {p}: {exc}")

    if not payloads:
        print(f"[aggregate] no JSONs in {results_dir}")
        return 1

    # Ordering heuristic — easy/median/medium/hard if present, else by label
    order = {"easy": 0, "median": 1, "medium": 2, "hard": 3}

    def _key(p: dict) -> tuple[int, str]:
        label = p.get("label", "")
        head = label.split("-", 1)[0]
        return (order.get(head, 99), label)

    payloads.sort(key=_key)

    header = (
        "| label | status | encode_s | peak_vram_mb | peak_rss_mb | cubes | dominant_stage |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows: list[str] = []
    fails: list[str] = []

    for p in payloads:
        label = p.get("label", "?")
        status = p.get("status", "?")
        enc = _fmt_float(p.get("wall_encode_s"))
        vram = _fmt_float(p.get("peak_vram_mb"), "{:.0f}")
        rss = _fmt_float(p.get("peak_host_rss_mb"), "{:.0f}")
        cubes = _fmt_int(p.get("cube_count"))
        dom = _dominant_stage(p.get("stage_walls_s") or {})
        rows.append(f"| {label} | {status} | {enc} | {vram} | {rss} | {cubes} | {dom} |")

        if not p.get("success", False):
            fails.append(
                f"- **{label}** — status=`{status}` "
                f"err=`{(p.get('error_msg') or '')[:200]}`"
            )

    lines = [
        "# OOM Validation Summary",
        "",
        f"Source: `{results_dir}`  (n={len(payloads)})",
        "",
        header.rstrip(),
    ]
    lines.extend(rows)
    if fails:
        lines.append("")
        lines.append("## Failures")
        lines.extend(fails)
    lines.append("")

    text = "\n".join(lines)
    summary_path.write_text(text)
    print(text)
    print(f"\n[aggregate] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
