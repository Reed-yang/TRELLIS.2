"""Produce coart/eval/golden_baseline.json with Layer V metrics per golden asset.

For helmet & triple_sphere: hardcoded from EXP-5 CSV.
For 6 val-split assets: left empty (reference line simply not drawn in wandb).

Downstream deep-eval code treats missing baseline gracefully.

Usage:
    .venv/bin/python scripts/coart_build_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")

# EXP-5 CSV direct values (res=512, Layer V; geometric_metrics.csv). triple_sphere
# uses the closest structural match from EXP-5: nested_spheres.
_EXP5_HARDCODED = {
    "helmet": {
        "cd": 1.2908307326142676e-05,
        "nc": 0.7730708122253418,
        "f_0.005": 0.8634670972824097,
        "f_0.001": 0.055654510855674744,
        "f_0.05": None,  # not in EXP-5 schema; filled to None for now
        "n_components": 208348,
        "euler": 4523,
        "n_boundary_edges": 146102,
        "is_watertight": 0.0,
    },
    "triple_sphere": {
        "cd": 1.2484781109378673e-05,
        "nc": 0.9995467066764832,
        "f_0.005": 0.8656343221664429,
        "f_0.001": 0.07622464001178741,
        "f_0.05": None,
        "n_components": 22,
        "euler": 5,
        "n_boundary_edges": 18,
        "is_watertight": 0.0,
    },
}


def main():
    asset_list_path = REPO / "coart" / "eval" / "golden_assets.json"
    if not asset_list_path.exists():
        print(f"[build_baseline] ERROR: {asset_list_path} missing. "
              f"Run scripts/coart_build_golden.py first.", file=sys.stderr)
        sys.exit(1)
    with open(asset_list_path) as fh:
        assets = json.load(fh)

    baseline = {}
    for a in assets:
        name = a["name"]
        if name in _EXP5_HARDCODED:
            baseline[name] = {"layer_v": dict(_EXP5_HARDCODED[name])}
            print(f"[build_baseline] {name}: using EXP-5 hardcoded Layer V")
        else:
            # val_p* assets: no baseline yet; wandb skips reference lines gracefully
            baseline[name] = {"layer_v": None}
            print(f"[build_baseline] {name}: no Layer V baseline "
                  f"(need to run custom pipeline offline; skipped)")

    out_path = REPO / "coart" / "eval" / "golden_baseline.json"
    with open(out_path, "w") as fh:
        json.dump(baseline, fh, indent=2)
    print(f"[build_baseline] wrote {out_path} ({len(baseline)} assets)")


if __name__ == "__main__":
    main()
