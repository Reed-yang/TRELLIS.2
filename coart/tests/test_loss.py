"""Unit tests for coart.vae.loss.compute_vae_loss."""
import torch

from coart.vae.loss import compute_vae_loss


def test_loss_dict_keys():
    pred = torch.randn(10, 18)
    target = torch.randn(10, 18)
    mu = torch.randn(10, 32)
    logvar = torch.randn(10, 32)
    subs_gt = [torch.randint(0, 2, (5, 8)).float() for _ in range(3)]
    subs = [torch.randn(5, 8) for _ in range(3)]

    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=subs_gt, subs=subs,
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    for k in ("total", "recon", "recon_p1", "recon_p2", "recon_ef", "kl", "subdiv"):
        assert k in d, f"missing key {k}"


def test_loss_recon_blocks_sum_weighted_to_total_recon():
    """recon_total == mean(mse across all 18 ch)."""
    pred = torch.randn(20, 18)
    target = torch.randn(20, 18)
    mu = torch.zeros(20, 32)
    logvar = torch.zeros(20, 32)
    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=[], subs=[],
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    expected_total_recon = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(d["recon"], expected_total_recon, atol=1e-6)


def test_loss_kl_zero_when_posterior_is_prior():
    pred = torch.zeros(5, 18)
    target = torch.zeros(5, 18)
    mu = torch.zeros(5, 32)
    logvar = torch.zeros(5, 32)  # exp(0)=1, (0+1-0-1)=0
    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=[], subs=[],
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    assert torch.allclose(d["kl"], torch.tensor(0.0), atol=1e-6)


def test_loss_subdiv_empty_when_no_levels():
    pred = torch.zeros(5, 18)
    target = torch.zeros(5, 18)
    mu = torch.zeros(5, 32)
    logvar = torch.zeros(5, 32)
    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=[], subs=[],
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    assert d["subdiv"].item() == 0.0
