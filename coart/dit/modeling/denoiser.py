"""CoartElasticSLatFlowModel — drop-in replacement for ElasticSLatFlowModel.

Replaces ``self.blocks`` with :class:`CoartDitBlock` so fused_modulation +
Coart RMSNorm are baked into the model graph (no monkey-patch). Inherits
everything else from upstream :class:`SLatFlowModel` (initialize_weights,
convert_to, forward, etc).

Class registration:
    Importing this module side-effect-registers
    :class:`CoartElasticSLatFlowModel` (and the non-elastic variant) onto
    ``trellis2.models`` so JSON configs with ``denoiser.name:
    "CoartElasticSLatFlowModel"`` resolve via the upstream BasicTrainer's
    ``getattr(models, name)`` lookup.
"""
from __future__ import annotations
from contextlib import contextmanager

from trellis2.models import structured_latent_flow as _slat_flow_mod
from trellis2.models.structured_latent_flow import SLatFlowModel
from trellis2.models.sparse_elastic_mixin import SparseTransformerElasticMixin

from .block import CoartDitBlock


@contextmanager
def _swap_block_class():
    """Temporarily redirect upstream's local ``ModulatedSparseTransformerCrossBlock``
    binding inside ``trellis2.models.structured_latent_flow`` to CoartDitBlock.

    SLatFlowModel.__init__ uses the module-local name, so this single rebinding
    causes all blocks to be constructed as CoartDitBlock — initialize_weights
    and convert_to(dtype) then run on Coart blocks directly. No state_dict
    copy / dtype round-trip is needed (which would not be bit-exact).
    """
    orig = _slat_flow_mod.ModulatedSparseTransformerCrossBlock
    _slat_flow_mod.ModulatedSparseTransformerCrossBlock = CoartDitBlock
    try:
        yield
    finally:
        _slat_flow_mod.ModulatedSparseTransformerCrossBlock = orig


class CoartSLatFlowModel(SLatFlowModel):
    """SLatFlowModel built with CoartDitBlock instead of upstream's
    ModulatedSparseTransformerCrossBlock.

    The swap happens via a local-binding override in
    ``trellis2.models.structured_latent_flow`` for the duration of the
    super().__init__ call, so all init / dtype-conversion logic runs on
    Coart blocks natively (no state_dict round-trip).
    """

    def __init__(self, **kwargs):
        with _swap_block_class():
            super().__init__(**kwargs)


class CoartElasticSLatFlowModel(SparseTransformerElasticMixin, CoartSLatFlowModel):
    """Coart SLat flow model with elastic memory management mixin."""
    pass


# Register into trellis2.models so JSON configs with
# "name": "CoartElasticSLatFlowModel" resolve via getattr(models, name).
from trellis2 import models as _trellis_models  # noqa: E402

_trellis_models.CoartElasticSLatFlowModel = CoartElasticSLatFlowModel
_trellis_models.CoartSLatFlowModel = CoartSLatFlowModel
