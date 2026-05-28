"""Compatibility exports for the split model implementation.

The original monolithic model implementation has been split into focused
subpackages under yffc.models. Importing from this module remains supported.
"""

from .common import _safe_tensor, _tukey_window, truncated_normal_
from .attention import Attention_block, CrossModalFusion, SpecularAwareAttention
from .blocks import (
    Conv2D_BN_activa,
    FeatureExtrationLayersB,
    MultiscaleConvBlock,
    TransposedConvBlock,
    TransposedConvBlock_same,
)
from .branches import AdaptiveSkipFusion, SwinEncoderBranch
from .ffc import FourierUnit, FFCResnetBlock, FFC_BN_ACT, SpectralTransform
from .heads import HighFrequencyEnhancement, ResidualRefinementHead
from .architecture import (
    Encoder,
    EnhancementLayers,
    GradientDecoder,
    MainDecoder,
    YFFC,
    _SpectralDecompositionUnit,
    cross_scale_consistency_loss,
)

__all__ = [name for name in globals() if not name.startswith('__')]
