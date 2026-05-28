"""Compatibility exports for the Y-FFC architecture modules."""

from .encoder import Encoder
from .decoders import GradientDecoder, MainDecoder
from .enhancement import EnhancementLayers, _SpectralDecompositionUnit
from .model import YFFC, cross_scale_consistency_loss

__all__ = [name for name in globals() if not name.startswith('__')]
