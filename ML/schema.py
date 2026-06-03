from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

FEATURE_SCHEMA_VERSION = "chan-ml-feature-v1"


@dataclass
class ModelManifest:
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    side: str = "both"
    model_kind: str = "unknown"
    threshold: float = 0.5
    feature_columns: List[str] = field(default_factory=list)
    label_config: Dict[str, Any] = field(default_factory=dict)
    train_range: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ModelManifest":
        return ModelManifest(
            feature_schema_version=str(
                payload.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
            ),
            side=str(payload.get("side", "both")),
            model_kind=str(payload.get("model_kind", "unknown")),
            threshold=float(payload.get("threshold", 0.5)),
            feature_columns=[str(col) for col in payload.get("feature_columns", [])],
            label_config=dict(payload.get("label_config", {}) or {}),
            train_range=dict(payload.get("train_range", {}) or {}),
            metrics=dict(payload.get("metrics", {}) or {}),
            notes=str(payload.get("notes", "")),
        )


def validate_manifest_features(
    manifest: ModelManifest,
    feature_columns: List[str],
) -> None:
    manifest_cols = list(manifest.feature_columns or [])
    if not manifest_cols:
        return
    expected = [str(col) for col in feature_columns]
    if manifest_cols != expected:
        missing = sorted(set(manifest_cols) - set(expected))
        extra = sorted(set(expected) - set(manifest_cols))
        raise ValueError(
            "model manifest feature columns do not match bundle: "
            f"missing={missing}, extra={extra}"
        )
