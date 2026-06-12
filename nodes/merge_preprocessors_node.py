import os
import shutil
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from nodes.feature_engineering_transformer import FeatureEngineeringTransformer


def copy_feature_engineering_transformer_to_temp(temp_dir: str) -> str:
    """
    Copy nodes/feature_engineering_transformer.py into temp_dir/nodes/
    so the downloaded ZIP has the same module path needed by joblib pickle.
    """

    source_path = Path(__file__).resolve().parent / "feature_engineering_transformer.py"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Feature engineering transformer file not found: {source_path}"
        )

    temp_nodes_dir = Path(temp_dir) / "nodes"
    temp_nodes_dir.mkdir(parents=True, exist_ok=True)

    init_path = temp_nodes_dir / "__init__.py"
    transformer_path = temp_nodes_dir / "feature_engineering_transformer.py"

    init_path.write_text("", encoding="utf-8")

    shutil.copyfile(source_path, transformer_path)

    return str(transformer_path)


def merge_preprocessors_node(state):

    if state.get("numerical_error"):
        return {"error": state.get("numerical_error")}

    if state.get("categorical_error"):
        return {"error": state.get("categorical_error")}

    temp_dir = state.get("temp_dir")

    if not temp_dir:
        return {"error": "Temporary directory not found."}

    os.makedirs(temp_dir, exist_ok=True)

    train_path = state.get("train_raw_path") or state["train_path"]
    test_path = state.get("test_raw_path") or state["test_path"]

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = state.get("target_column")

    # -------------------------
    # Separate X and y
    # -------------------------

    if target:
        X_train = train.drop(columns=[target])
        X_test = test.drop(columns=[target])

        y_train = train[target]
        y_test = test[target]

    else:
        X_train = train
        X_test = test

        y_train = None
        y_test = None

    num_cols = state.get("numerical_columns") or []
    cat_cols = state.get("categorical_columns") or []

    num_path = state.get("numerical_pipeline_path")
    cat_path = state.get("categorical_pipeline_path")

    transformers = []

    if num_cols:
        if not num_path or not os.path.exists(num_path):
            return {"error": "Numerical preprocessing pipeline not found."}

        numerical_pipeline = joblib.load(num_path)
        transformers.append(("num", numerical_pipeline, num_cols))

    if cat_cols:
        if not cat_path or not os.path.exists(cat_path):
            return {"error": "Categorical preprocessing pipeline not found."}

        categorical_pipeline = joblib.load(cat_path)
        transformers.append(("cat", categorical_pipeline, cat_cols))

    if not transformers:
        return {"error": "No numerical or categorical preprocessing pipelines found."}

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=True
    )

    # -------------------------
    # Build full pipeline
    # -------------------------

    steps = []

    feature_plan = state.get("feature_engineering_plan")

    if feature_plan and state.get("feature_engineering_approved") is True:
        steps.append(
            (
                "feature_engineering",
                FeatureEngineeringTransformer(
                    plan=feature_plan,
                    target_column=target
                )
            )
        )

    steps.append(("preprocessor", preprocessor))

    pipeline = Pipeline(steps)

    # -------------------------
    # Fit transform X
    # -------------------------

    X_train_clean = X_train.replace({pd.NA: np.nan})
    X_test_clean = X_test.replace({pd.NA: np.nan})

    train_array = pipeline.fit_transform(X_train_clean, y_train)
    test_array = pipeline.transform(X_test_clean)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    train_processed = pd.DataFrame(
        train_array,
        columns=feature_names
    )

    test_processed = pd.DataFrame(
        test_array,
        columns=feature_names
    )

    # -------------------------
    # Encode target if categorical
    # -------------------------

    target_encoder_path = None

    if target:

        if y_train.dtype == "object" or y_train.nunique() < 20:
            encoder = LabelEncoder()

            y_train = encoder.fit_transform(y_train.astype(str))
            y_test = encoder.transform(y_test.astype(str))

            target_encoder_path = os.path.join(temp_dir, "target_encoder.pkl")
            joblib.dump(encoder, target_encoder_path)

        train_processed[target] = y_train
        test_processed[target] = y_test

    # -------------------------
    # Save processed files
    # -------------------------

    train_out = os.path.join(temp_dir, "processed_train.csv")
    test_out = os.path.join(temp_dir, "processed_test.csv")

    train_processed.to_csv(train_out, index=False)
    test_processed.to_csv(test_out, index=False)

    # -------------------------
    # Save final preprocessor
    # -------------------------

    preprocessor_path = os.path.join(temp_dir, "preprocessor.pkl")
    joblib.dump(pipeline, preprocessor_path)

    # -------------------------
    # Copy custom transformer file for ZIP
    # -------------------------

    feature_engineering_transformer_path = copy_feature_engineering_transformer_to_temp(
        temp_dir
    )

    return {
        "processed_train_path": train_out,
        "processed_test_path": test_out,

        "final_preprocessor_path": preprocessor_path,

        "feature_engineering_transformer_path": feature_engineering_transformer_path,

        "target_encoder_path": target_encoder_path,

        "steps": state.get("steps", []) + [
            "Numerical preprocessing ready.",
            "Categorical preprocessing ready.",
            "Merged preprocessors with feature engineering.",
            "Processed train/test saved.",
            "Final preprocessor saved.",
            "Feature engineering transformer file copied for export.",
        ],

        "error": None,
    }