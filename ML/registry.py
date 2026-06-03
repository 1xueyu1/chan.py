from __future__ import annotations

import json
from pathlib import Path

from .schema import ModelManifest, validate_manifest_features

MANIFEST_SUFFIX = ".manifest.json"


def manifest_path_for_model(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + MANIFEST_SUFFIX)


def save_manifest(model_path: str | Path, manifest: ModelManifest) -> Path:
    path = manifest_path_for_model(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_manifest(model_path: str | Path) -> ModelManifest | None:
    path = manifest_path_for_model(model_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelManifest.from_dict(payload)


def validate_model_manifest(model_path: str | Path, feature_columns: list[str]) -> None:
    manifest = load_manifest(model_path)
    if manifest is None:
        return
    validate_manifest_features(manifest, feature_columns)
