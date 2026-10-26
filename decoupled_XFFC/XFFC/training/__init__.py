"""Training helpers for the decoupled Y-FFC copy."""

from .engine import train_model
from .loops import (
    _format_loss_diag,
    _repair_nonfinite_model,
    _sanitize_optimizer_state,
    _tensors_are_finite,
    _tta_forward,
    train_one_epoch,
    validate,
)
from .metrics import MetricTracker, SUB_LOSS_DISPLAY_NAMES, SUB_LOSS_KEYS
from .optimization import ImprovedWeightScheduler, build_optimizer_v2, format_lr_groups
from .reporting import print_epoch_log, save_training_summary
from .simple_loop import (
    train as train_legacy,
    train_one_epoch as train_one_epoch_legacy,
    validate as validate_legacy,
)
from .utils import (
    DynamicLossWeightScheduler,
    EMA,
    build_optimizer,
    build_scheduler,
    replace_bn_with_gn,
)
from .visualization import (
    build_fixed_visualization_loader,
    plot_individual_curves,
    plot_sub_loss_history,
    plot_training_curves_realtime,
    plot_training_history,
    visualize_sample_during_training,
)

__all__ = [
    "DynamicLossWeightScheduler",
    "EMA",
    "ImprovedWeightScheduler",
    "MetricTracker",
    "SUB_LOSS_DISPLAY_NAMES",
    "SUB_LOSS_KEYS",
    "_format_loss_diag",
    "_repair_nonfinite_model",
    "_sanitize_optimizer_state",
    "_tensors_are_finite",
    "_tta_forward",
    "build_fixed_visualization_loader",
    "build_optimizer",
    "build_optimizer_v2",
    "build_scheduler",
    "format_lr_groups",
    "plot_individual_curves",
    "plot_sub_loss_history",
    "plot_training_curves_realtime",
    "plot_training_history",
    "print_epoch_log",
    "replace_bn_with_gn",
    "save_training_summary",
    "train_legacy",
    "train_model",
    "train_one_epoch",
    "train_one_epoch_legacy",
    "validate",
    "validate_legacy",
    "visualize_sample_during_training",
]
