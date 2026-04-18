"""For each top-20 entry, grep corep_fast/ for its frame_name leaf and return
candidate source locations. Classification is human-judged and appended to
logs/findings_sync_sources.md.

Usage: python -m tmp.sync_spike.classify_top20
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOP20 = REPO_ROOT / "tmp/sync_spike/top20_d2h.json"


def leaf_name(frame_name: str) -> str:
    """Extract the innermost identifier from a frame name like
    'corep_fast/stages/s6_collapse.py(123): _build_something'."""
    m = re.search(r":\s*(\w+)\s*$", frame_name)
    if m:
        return m.group(1)
    m = re.search(r"(\w+)\s*$", frame_name)
    return m.group(1) if m else frame_name


def grep(term: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["grep", "-rn", "--include=*.py", term, "corep_fast/"],
            cwd=REPO_ROOT, text=True
        )
    except subprocess.CalledProcessError:
        return []
    return out.splitlines()[:10]


def main():
    if not TOP20.exists():
        sys.exit(f"Missing {TOP20} — run Task 5 first.")
    rows = json.loads(TOP20.read_text())
    print(f"# Top-20 D2H → source candidates\n")
    for r in rows:
        leaf = leaf_name(r["frame_name"])
        hits = grep(leaf)
        print(f"## rank {r['rank']}: stage={r['stage']}  count={r['count']}  "
              f"total_ms={r['total_dur_ms']:.2f}")
        print(f"frame: {r['frame_name']}")
        print(f"leaf: {leaf}")
        if hits:
            print("grep candidates:")
            for h in hits[:5]:
                print(f"  {h}")
        else:
            print("grep candidates: <none>")
        print()


if __name__ == "__main__":
    main()
