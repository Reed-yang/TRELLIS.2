"""Probe nsys SQLite trace for the longest cudaDeviceSynchronize.

Deep Profiling (2026-04-16/17) reported 20 cudaDeviceSynchronize calls totaling
1049ms, with one call ~1048ms. This script locates that call's timestamp so we
can cross-reference the Python stack in torch.profiler trace.

Usage: python -m tmp.sync_spike.probe_nsys_sync
"""
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SQLITE_PATH = REPO_ROOT / "tmp/profile_deep/results/nsys_res256_full.sqlite"


def find_runtime_table(con: sqlite3.Connection) -> str:
    """Return the table name that holds CUDA runtime API events."""
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name LIKE '%CUPTI_ACTIVITY_KIND_RUNTIME%' OR name LIKE '%CUDA_API%')"
    )
    rows = [r[0] for r in cur.fetchall()]
    if not rows:
        raise RuntimeError(
            "Could not find runtime API table in nsys SQLite. "
            "Run `sqlite3 <db> .tables` and inspect."
        )
    return rows[0]


def find_devicesync_calls(con: sqlite3.Connection, table: str, top_n: int = 5):
    """Return the top-N longest cudaDeviceSynchronize calls.

    nsys stores the kernel / API name in a `StringIds` lookup table;
    `CUPTI_ACTIVITY_KIND_RUNTIME` has a `nameId` FK.
    """
    # Find the string ID for cudaDeviceSynchronize
    cur = con.execute(
        "SELECT id FROM StringIds WHERE value = 'cudaDeviceSynchronize_v3020' "
        "OR value = 'cudaDeviceSynchronize'"
    )
    ids = [r[0] for r in cur.fetchall()]
    if not ids:
        raise RuntimeError("cudaDeviceSynchronize not found in StringIds")

    placeholders = ",".join("?" * len(ids))
    q = (
        f"SELECT start, end, (end-start) AS dur_ns, correlationId, globalTid "
        f"FROM {table} "
        f"WHERE nameId IN ({placeholders}) "
        f"ORDER BY dur_ns DESC LIMIT {top_n}"
    )
    cur = con.execute(q, ids)
    return cur.fetchall()


def main():
    if not SQLITE_PATH.exists():
        sys.exit(f"Missing nsys SQLite: {SQLITE_PATH}")
    with sqlite3.connect(str(SQLITE_PATH)) as con:
        table = find_runtime_table(con)
        print(f"[info] runtime API table: {table}")

        calls = find_devicesync_calls(con, table, top_n=5)
        if not calls:
            sys.exit("No cudaDeviceSynchronize calls found.")

        print(f"\nTop 5 longest cudaDeviceSynchronize calls:")
        print(f"{'rank':<5}{'start_ns':>16}{'end_ns':>16}{'dur_ms':>12}{'corrId':>10}{'tid':>20}")
        for rank, (start, end, dur_ns, corr, tid) in enumerate(calls, 1):
            print(f"{rank:<5}{start:>16}{end:>16}{dur_ns/1e6:>12.3f}{corr:>10}{tid:>20}")

        top = calls[0]
        assert top[2] / 1e6 > 500, (
            f"Longest cudaDeviceSynchronize is {top[2]/1e6:.1f}ms, expected >500ms "
            f"per Deep Profiling finding. Data may have drifted."
        )
        print(f"\n[ok] longest cudaDeviceSynchronize: {top[2]/1e6:.2f}ms "
              f"(start={top[0]}, end={top[1]})")

    # Save the top call to a small JSON for Task 3.
    import json
    out_path = REPO_ROOT / "tmp/sync_spike/nsys_top_devicesync.json"
    with open(out_path, "w") as f:
        json.dump({
            "start_ns": top[0],
            "end_ns": top[1],
            "dur_ns": top[2],
            "dur_ms": top[2] / 1e6,
            "correlationId": top[3],
            "globalTid": top[4],
        }, f, indent=2)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
