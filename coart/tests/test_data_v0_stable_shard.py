"""Test the inline stable_shard helper used by dino/slat scripts."""
import importlib.util, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_shard_partition_is_complete_and_disjoint():
    cache_dino = _load("coart_cache_dino")
    sha_pool = [f"{i:08x}" + "0" * 56 for i in range(1000)]
    world = 8
    assignments = [[s for s in sha_pool if cache_dino.stable_shard(s, world, r)] for r in range(world)]
    flat = [s for row in assignments for s in row]
    assert len(flat) == 1000
    assert len(set(flat)) == 1000
    sizes = [len(row) for row in assignments]
    assert max(sizes) - min(sizes) < 0.3 * 1000 / world
