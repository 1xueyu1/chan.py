import numpy as np
import pandas as pd

from ML.model import ModelBundle
from ML.registry import load_manifest, manifest_path_for_model


class DummyEstimator:
    def predict_proba(self, frame):
        return np.column_stack(
            [np.zeros(len(frame), dtype="float64"), np.ones(len(frame)) * 0.7]
        )


def test_model_bundle_save_writes_manifest(tmp_path):
    path = tmp_path / "model.pkl"
    bundle = ModelBundle(
        estimator=DummyEstimator(),
        feature_columns=["a", "b"],
        fill_values={"a": 0.0, "b": 0.0},
        threshold=0.6,
        side="buy",
        model_kind="dummy",
    )

    bundle.save(path)
    loaded = ModelBundle.load(path)
    manifest = load_manifest(path)

    assert path.exists()
    assert manifest_path_for_model(path).exists()
    assert loaded.predict_proba(pd.DataFrame({"a": [1.0], "b": [2.0]}))[0] == 0.7
    assert manifest is not None
    assert manifest.side == "buy"
    assert manifest.model_kind == "dummy"
    assert manifest.feature_columns == ["a", "b"]
