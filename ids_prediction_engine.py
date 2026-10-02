"""Inference helpers for the UNSW-NB15 29-feature, two-stage IDS bundle.

The bundle is exported by the packaging cell in the companion notebook.
Input DataFrames may contain the 29 retained features or all 34 raw dataset
features; the five dropped fields are ignored when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load the versioned joblib bundle created by the Kaggle export cell."""
    bundle = joblib.load(path)
    required = {
        "binary_model",
        "binary_encoder",
        "multiclass_model",
        "multiclass_preprocessor",
        "metadata",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"IDS bundle is missing entries: {sorted(missing)}")
    if bundle["metadata"].get("bundle_format") != "unsw_nb15_ids_29feature_v1":
        raise ValueError("Unsupported or missing IDS bundle format marker.")
    return bundle


def predict_ids(input_df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Run Stage 1 (Normal/Attack), then Stage 2 on predicted attacks.

    Returns one row per input sample with binary confidence, attack risk, and
    attack category confidence. Rows predicted Normal have no attack category.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame")

    metadata = bundle["metadata"]
    required_features = metadata["required_input_features"]
    missing = [name for name in required_features if name not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required 29-feature input columns: {missing}")

    # Select the exact raw feature set and order used by the trained models.
    x = input_df.loc[:, required_features].copy()
    cat_features = metadata["categorical_features"]
    numeric_features = metadata["binary_numeric_features"]

    encoded_cat = bundle["binary_encoder"].transform(x.loc[:, cat_features])
    encoded_binary = np.hstack([x.loc[:, numeric_features].to_numpy(), encoded_cat])
    expected_binary = metadata["binary_encoded_dimension"]
    if encoded_binary.shape[1] != expected_binary:
        raise ValueError(
            f"Stage 1 encoded {encoded_binary.shape[1]} columns; expected {expected_binary}."
        )

    binary_model = bundle["binary_model"]
    binary_pred = binary_model.predict(encoded_binary)
    binary_classes = list(binary_model.classes_)
    if 0 not in binary_classes or 1 not in binary_classes:
        raise ValueError(f"Stage 1 expected classes 0/1, got {binary_classes}.")
    binary_prob = binary_model.predict_proba(encoded_binary)
    normal_col = binary_classes.index(0)
    attack_col = binary_classes.index(1)
    attack_risk = binary_prob[:, attack_col]
    predicted_confidence = binary_prob[
        np.arange(len(binary_pred)),
        [binary_classes.index(label) for label in binary_pred],
    ]

    results: list[dict[str, Any]] = []
    attack_positions = np.flatnonzero(binary_pred == 1)
    attack_labels: dict[int, Any] = {}
    attack_confidences: dict[int, float] = {}

    if len(attack_positions):
        mc_features = metadata["multiclass_features"]
        encoded_mc = bundle["multiclass_preprocessor"].transform(
            x.iloc[attack_positions].loc[:, mc_features]
        )
        expected_mc = metadata["multiclass_encoded_dimension"]
        if encoded_mc.shape[1] != expected_mc:
            raise ValueError(
                f"Stage 2 encoded {encoded_mc.shape[1]} columns; expected {expected_mc}."
            )

        mc_model = bundle["multiclass_model"]
        mc_pred = mc_model.predict(encoded_mc)
        mc_prob = mc_model.predict_proba(encoded_mc)
        mc_classes = list(mc_model.classes_)
        for j, position in enumerate(attack_positions):
            label = mc_pred[j]
            attack_labels[int(position)] = label
            attack_confidences[int(position)] = float(
                mc_prob[j, mc_classes.index(label)]
            )

    for i, label in enumerate(binary_pred):
        is_attack = label == 1
        results.append(
            {
                "prediction": "Attack" if is_attack else "Normal",
                "attack_type": attack_labels.get(i),
                "binary_prediction": int(label),
                "binary_confidence": float(predicted_confidence[i]),
                "attack_probability": float(attack_risk[i]),
                "attack_confidence": attack_confidences.get(i),
            }
        )

    return pd.DataFrame(results, index=input_df.index)
