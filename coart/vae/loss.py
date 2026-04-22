"""Block-decomposed VAE loss for 18-ch corep feat18.

Returns a dict with the total loss (for backward) and separate block components
(for TB logging). The three recon blocks correspond to the channel layout:
    ch 0:3   (p1)  - point1 xyz
    ch 3:6   (p2)  - point2 xyz
    ch 6:18  (ef)  - edge/face weights

NO render loss. The loss total is:
    loss = recon + lambda_kl * kl + lambda_subdiv * subdiv
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def compute_vae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    subs_gt: List[torch.Tensor],
    subs: List[torch.Tensor],
    lambda_kl: float = 1e-6,
    lambda_subdiv: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute total + per-block VAE loss.

    Args:
        pred, target: (N, 18) feature tensors in normalised space.
        mu, logvar: (N, latent_channels) posterior params.
        subs_gt: list of ground-truth subdivision bitmaps per decoder level.
        subs: list of logits predicted for each subdivision level.
        lambda_kl, lambda_subdiv: scalar weights.

    Returns:
        dict with keys: total, recon, recon_p1, recon_p2, recon_ef, kl, subdiv.
        Every value is a 0-d tensor on pred's device.
    """
    pred_f = pred.float()
    target_f = target.float()
    mu_f = mu.float()
    logvar_f = logvar.float()

    recon_total = F.mse_loss(pred_f, target_f)
    recon_p1 = F.mse_loss(pred_f[:, 0:3], target_f[:, 0:3])
    recon_p2 = F.mse_loss(pred_f[:, 3:6], target_f[:, 3:6])
    recon_ef = F.mse_loss(pred_f[:, 6:18], target_f[:, 6:18])

    kl = 0.5 * torch.mean(mu_f.pow(2) + logvar_f.exp() - logvar_f - 1)

    if len(subs) > 0:
        subdiv = sum(
            F.binary_cross_entropy_with_logits(s.float(), g.float())
            for s, g in zip(subs, subs_gt)
        ) / len(subs)
    else:
        subdiv = torch.zeros((), device=pred.device)

    total = recon_total + lambda_kl * kl + lambda_subdiv * subdiv

    return {
        "total": total,
        "recon": recon_total,
        "recon_p1": recon_p1,
        "recon_p2": recon_p2,
        "recon_ef": recon_ef,
        "kl": kl,
        "subdiv": subdiv,
    }
