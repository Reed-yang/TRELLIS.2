"""Unit tests for coart.common.ema.EMAModel."""
import copy

import pytest
import torch
import torch.nn as nn

from coart.common.ema import EMAModel


def _tiny_model():
    return nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))


def test_ema_initial_shadow_matches_model():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.999)
    for p, s in zip(m.parameters(), ema.shadow_params()):
        assert torch.allclose(p.detach().float(), s), "initial shadow must match params"


def test_ema_update_moves_toward_online():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.9)
    snap = [s.clone() for s in ema.shadow_params()]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p))
    ema.update(m)
    for old, s in zip(snap, ema.shadow_params()):
        assert torch.allclose(s, old + 0.1, atol=1e-6), "shadow must drift by (1-decay) * delta"


def test_ema_state_dict_roundtrip():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.999)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p) * 0.5)
    ema.update(m)
    sd = ema.state_dict()
    ema2 = EMAModel(_tiny_model(), decay=0.999)
    ema2.load_state_dict(sd)
    for s1, s2 in zip(ema.shadow_params(), ema2.shadow_params()):
        assert torch.allclose(s1, s2)


def test_ema_copy_to_writes_shadow_into_target():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.9)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(99.0)
    target = _tiny_model()
    ema.copy_to(target)
    for s, p in zip(ema.shadow_params(), target.parameters()):
        assert torch.allclose(s, p.detach().float())


def test_ema_decay_bounds():
    m = _tiny_model()
    with pytest.raises(ValueError):
        EMAModel(m, decay=1.5)
    with pytest.raises(ValueError):
        EMAModel(m, decay=-0.1)
