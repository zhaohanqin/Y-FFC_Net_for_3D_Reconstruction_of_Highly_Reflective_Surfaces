"""Training helpers for the decoupled Y-FFC copy."""

from .simple_loop import train, train_one_epoch, validate
from .utils import (
    DynamicLossWeightScheduler,
    EMA,
    build_optimizer,
    build_scheduler,
    replace_bn_with_gn,
)

_ENGINE_EXPORTS = {
    "ImprovedWeightScheduler",
    "MetricTracker",
    "SUB_LOSS_DISPLAY_NAMES",
    "SUB_LOSS_KEYS",
    "_tta_forward",
    "build_optimizer_v2",
    "plot_individual_curves",
    "plot_sub_loss_history",
    "plot_training_curves_realtime",
    "plot_training_history",
    "print_epoch_log",
    "save_training_summary",
    "train_model",
    "train_one_epoch_enhanced",
    "validate_enhanced",
    "visualize_sample_during_training",
}


def __getattr__(name):
    if name not in _ENGINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {"ImprovedWeightScheduler", "build_optimizer_v2"}:
        from . import optimization
        return getattr(optimization, name)
    if name in {"MetricTracker", "SUB_LOSS_DISPLAY_NAMES", "SUB_LOSS_KEYS"}:
        from . import metrics
        return getattr(metrics, name)
    if name in {"_tta_forward", "train_one_epoch_enhanced", "validate_enhanced"}:
        from . import loops
        if name == "train_one_epoch_enhanced":
            return loops.train_one_epoch
        if name == "validate_enhanced":
            return loops.validate
        return getattr(loops, name)
    if name in {"plot_individual_curves", "plot_sub_loss_history", "plot_training_curves_realtime", "plot_training_history", "visualize_sample_during_training"}:
        from . import visualization
        return getattr(visualization, name)
    if name in {"print_epoch_log", "save_training_summary"}:
        from . import reporting
        return getattr(reporting, name)
    from .engine import train_model
    return train_model


__all__ = [
    "DynamicLossWeightScheduler",
    "EMA",
    "build_optimizer",
    "build_scheduler",
    "replace_bn_with_gn",
    "train",
    "train_one_epoch",
    "validate",
    *_ENGINE_EXPORTS,
]
