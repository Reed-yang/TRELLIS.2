import sys
sys.path.insert(0, "tmp/cpu_profile")
from t0_driver import maybe_force_serial, patch_stage_pools, SerialPool
maybe_force_serial()

from multiprocessing import Pool as _Pool
print("after maybe_force_serial, from-import Pool is SerialPool:", _Pool is SerialPool)

from corep_fast.pipeline import corep_pipeline
patch_stage_pools()

for mod_name in (
    "corep_fast.stages.s4_face_point",
    "corep_fast.stages.s6_collapse",
    "corep_fast.stages.s7_rank_assign",
    "corep_fast.stages.s8_collapse",
):
    mod = sys.modules.get(mod_name)
    if mod is None:
        print(f"MISSING {mod_name}")
        continue
    found = []
    for alias in ("_Pool", "Pool"):
        if hasattr(mod, alias):
            v = getattr(mod, alias)
            name = getattr(v, "__name__", repr(v))
            is_s = (v is SerialPool)
            found.append(f"{alias}={name}(serial={is_s})")
    if not found:
        print(f"{mod_name}: NO module-level _Pool/Pool alias (import is inside function)")
    else:
        print(f"{mod_name}: {found}")
