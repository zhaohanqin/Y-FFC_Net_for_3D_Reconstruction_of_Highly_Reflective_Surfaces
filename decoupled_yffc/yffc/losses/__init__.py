"""Loss functions and image quality metrics."""

from .metrics import MSSSIM, SSIM, gaussian_window
from .basic import (
    PerceptualLoss,
    _as_batched_4d,
    _restore_tensor_dim,
    _safe_tensor,
    charbonnier_loss,
    compute_edge_weight_map,
    compute_highlight_prior,
    compute_input_gradient_target,
    compute_input_gradient_weight_map,
    normalize_per_sample,
    normalized_sobel_magnitude,
    sobel_magnitude,
    statistical_loss,
)
from .specular import (
    GradientConsistencyLoss,
    SpecularDiceLoss,
    SpecularFocalLoss,
    specular_ohem_loss,
)
from .composite import CompositeLoss, EnhancedCompositeLoss

__all__ = [name for name in globals() if not name.startswith('__')]
