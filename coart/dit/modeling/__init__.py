"""coart.dit.modeling — model components.

Importing this package side-effect-registers
:class:`CoartElasticSLatFlowModel` (and its non-elastic variant) into
``trellis2.models`` so JSON configs may reference
``"name": "CoartElasticSLatFlowModel"``.
"""
from . import denoiser  # noqa: F401  (registers into trellis2.models)
from .block import CoartDitBlock  # noqa: F401
from .rmsnorm import CoartSparseMultiHeadRMSNorm  # noqa: F401
from .denoiser import CoartElasticSLatFlowModel, CoartSLatFlowModel  # noqa: F401

__all__ = [
    "CoartDitBlock",
    "CoartSparseMultiHeadRMSNorm",
    "CoartElasticSLatFlowModel",
    "CoartSLatFlowModel",
]
