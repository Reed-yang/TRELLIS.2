"""Grep corep_fast/ for known implicit and explicit CUDA sync triggers.

Sync triggers targeted:
  - Explicit: torch.cuda.synchronize, torch.cuda.current_stream().synchronize
  - Implicit: .item(), .cpu(), .tolist(), .numpy()
  - Boolean-bridge: `if <tensor>:`, `while <tensor>:`, `bool(<tensor>)`

Output: tmp/sync_spike/grep_sync_sources.json — list of {file, line, pattern, snippet}

Usage: python -m tmp.sync_spike.grep_sync_sources
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COREP_FAST = REPO_ROOT / "corep_fast"

PATTERNS = [
    ("explicit_sync",    re.compile(r"torch\.cuda\.synchronize\s*\(")),
    ("stream_sync",      re.compile(r"\.stream\(\)\.synchronize\s*\(|current_stream\(\)\.synchronize\s*\(")),
    ("item",             re.compile(r"\.item\s*\(\s*\)")),
    ("cpu_call",         re.compile(r"\.cpu\s*\(\s*\)")),
    ("tolist",           re.compile(r"\.tolist\s*\(\s*\)")),
    ("numpy_call",       re.compile(r"\.numpy\s*\(\s*\)")),
    ("bool_bridge",      re.compile(r"(?<![a-zA-Z_])bool\s*\(")),
]

def scan_file(path: Path) -> list[dict]:
    hits = []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return hits
    for lineno, line in enumerate(lines, 1):
        for label, rx in PATTERNS:
            if rx.search(line):
                hits.append({
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": lineno,
                    "pattern": label,
                    "snippet": line.strip(),
                })
    return hits


def main():
    if not COREP_FAST.exists():
        sys.exit(f"Missing {COREP_FAST}")
    all_hits = []
    for p in COREP_FAST.rglob("*.py"):
        all_hits.extend(scan_file(p))

    by_pattern: dict[str, int] = {}
    for h in all_hits:
        by_pattern[h["pattern"]] = by_pattern.get(h["pattern"], 0) + 1

    print(f"Found {len(all_hits)} total matches across {len(set(h['file'] for h in all_hits))} files:")
    for label, cnt in sorted(by_pattern.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<20} {cnt}")

    out_path = REPO_ROOT / "tmp/sync_spike/grep_sync_sources.json"
    with open(out_path, "w") as f:
        json.dump(all_hits, f, indent=2)
    print(f"\n[write] {out_path}  (all hits)")


if __name__ == "__main__":
    main()
