"""Fast Chan-based machine learning utilities."""

from .dataset import build_dataset, build_symbol_dataset
from .features import build_event_features, infer_feature_columns
from .label import LabelConfig, label_events
from .model import ModelBundle, train_classifier

__all__ = [
    "LabelConfig",
    "ModelBundle",
    "build_dataset",
    "build_event_features",
    "build_symbol_dataset",
    "infer_feature_columns",
    "label_events",
    "train_classifier",
]
