import sys
import tempfile
from pathlib import Path
sys.path.insert(0, "tmp/cpu_profile")
from t0_driver import maybe_force_serial, patch_stage_pools, SerialPool, make_fixture
maybe_force_serial()

import torch
from corep_fast.pipeline import corep_pipeline

device = torch.device("cuda:0")
with tempfile.TemporaryDirectory() as tmp:
    mesh_path = str(Path(tmp) / "fixture.ply")
    make_fixture(mesh_path)
    # tiny warmup so all stage modules load
    _ = corep_pipeline(mesh_path, 64, device)
    torch.cuda.synchronize()

# Now all stages should be loaded
for mod_name in (
    "corep_fast.stages.s4_face_point",
    "corep_fast.stages.s6_collapse",
    "corep_fast.stages.s7_rank_assign",
    "corep_fast.stages.s8_collapse",
):
    mod = sys.modules.get(mod_name)
    if mod is None:
        print(f"MISSING (still) {mod_name}")
        continue
    found = []
    for alias in ("_Pool", "Pool"):
        if hasattr(mod, alias):
            v = getattr(mod, alias)
            name = getattr(v, "__name__", repr(v))
            is_s = (v is SerialPool)
            found.append(f"{alias}={name}(serial={is_s})")
    if not found:
        print(f"{mod_name}: NO module-level alias (function-scoped import only) -- OK because mp.Pool is patched")
    else:
        print(f"{mod_name}: {found}")

# Verify mp.Pool is still SerialPool
import multiprocessing as mp
print(f"mp.Pool is SerialPool: {mp.Pool is SerialPool}")
