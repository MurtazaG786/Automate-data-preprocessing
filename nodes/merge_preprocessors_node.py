import os
import shutil
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from google import genai
from google.genai import types

from nodes.feature_engineering_transformer import FeatureEngineeringTransformer
from nodes.llm_env import get_primary_api_key_model


class SMOTEPlan(BaseModel):
    apply_smote: bool = Field(
        default=False,
        description="Whether SMOTE should be applied to the training set."
    )
    reason: str = Field(
        default="",
        description="Short explanation for the SMOTE decision."
    )


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
    problem_type = state.get("problem_type")

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

    smote_applied = False
    smote_reason = state.get("smote_reason") or ""

    # -------------------------
    # Decide whether to apply SMOTE
    # -------------------------

    if target and state.get("problem_type") == "classification":
        class_counts = y_train.value_counts(dropna=False)
        minority_count = int(class_counts.min()) if not class_counts.empty else 0
        majority_count = int(class_counts.max()) if not class_counts.empty else 0
        imbalance_ratio = (minority_count / majority_count) if majority_count else 0.0

        target_balance_info = (
            f"Target column: {target}\n"
            f"Class counts: {class_counts.to_dict()}\n"
            f"Minority/majority ratio: {imbalance_ratio:.3f}\n"
            f"Minority samples: {minority_count}\n"
            "Use SMOTE only for clear classification imbalance and only when the minority class has enough samples."
        )

        api_key, model_name = get_primary_api_key_model()

        smote_requested = False
        if api_key and model_name:
            client = genai.Client(api_key=api_key)

            prompt = f"""
You are a machine learning preprocessing expert.

Decide whether SMOTE should be applied to the training set after feature preprocessing.

Rules:
- Apply SMOTE only for supervised classification problems.
- Do not apply SMOTE for regression or unsupervised problems.
- Do not apply SMOTE if the classes are already reasonably balanced.
- Do not apply SMOTE if the minority class is too small to oversample safely.

Dataset summary:
{target_balance_info}

Current feature matrix shape:
{train_processed.shape}

Return JSON with:
- apply_smote: true/false
- reason: short explanation for the decision
"""

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SMOTEPlan,
                    ),
                )
                smote_plan = response.parsed
                smote_requested = bool(smote_plan.apply_smote)
                smote_reason = smote_plan.reason or smote_reason
            except Exception:
                smote_requested = bool(state.get("apply_smote"))
        else:
            smote_requested = bool(state.get("apply_smote"))

        if smote_requested and minority_count >= 2 and len(class_counts) >= 2:
            k_neighbors = min(5, minority_count - 1)
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_resampled, y_resampled = sampler.fit_resample(train_processed, y_train)

            train_processed = pd.DataFrame(X_resampled, columns=feature_names)
            y_train = pd.Series(y_resampled, name=target)
            smote_applied = True
            smote_reason = smote_reason or (
                f"SMOTE applied with k_neighbors={k_neighbors} to rebalance the training target."
            )
        elif smote_requested:
            smote_reason = smote_reason or "SMOTE skipped because the minority class is too small or only one class is present."

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
        "smote_applied": smote_applied,
        "smote_reason": smote_reason,

        "steps": state.get("steps", []) + [
            "Numerical preprocessing ready.",
            "Categorical preprocessing ready.",
            "Merged preprocessors with feature engineering.",
            ("Applied SMOTE to the training set." if smote_applied else "SMOTE not applied."),
            "Processed train/test saved.",
            "Final preprocessor saved.",
            "Feature engineering transformer file copied for export.",
        ],

        "error": None,
    }