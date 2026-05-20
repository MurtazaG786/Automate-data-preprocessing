import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from typing import Literal
from pydantic import BaseModel

from google import genai
from google.genai import types
from llm_config import build_fallback_llm
class NumericalColumnPlan(BaseModel):
    name: str


    impute_strategy: Literal["mean", "median", "most_frequent", "none"]
    # none   → skip imputation (column has no missing values)
    # mean   → normal distribution
    # median → skewed distribution or has outliers
    # most_frequent → discrete numerical values

    outlier_handling: Literal["iqr", "zscore", "clip", "none"]
    # none   → column is already clean
    # iqr    → moderately skewed
    # zscore → normally distributed
    # clip   → heavy outliers, cap at percentile

    scale: Literal["standard", "minmax", "robust", "none"]
    # none     → already scaled or tree-based model (no scaling needed)
    # standard → normal distribution
    # minmax   → bounded data [0-100]
    # robust   → outliers remain after handling

# Use Literal["...", "none"] for everything — because:

# LLM outputs strings, not Python None
# Easier to read in JSON response
# No null checks needed in code — just if plan.scale != "none"
# Consistent across all fields

class NumericalPreprocessingPlan(BaseModel):
    columns: list[NumericalColumnPlan]


def numerical_preprocessing_node(state):

    train = pd.read_csv(state["train_path"])
    cols = state["numerical_columns"]

    # -------------------------
    # Summary stats
    # -------------------------
    summary = train[cols].describe().T
    summary["missing_percent"] = train[cols].isna().mean() * 100
    summary["skew"] = train[cols].skew()
    summary["unique"] = train[cols].nunique()

    # -------------------------
    # LLM setup
    # -------------------------
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not api_key or not model_name:
        return {
            "error":
            "Missing GOOGLE_API_KEY or MODEL_NAME"
        }
    llm=build_fallback_llm()

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an ML preprocessing expert.

For each numerical column decide:

- impute: true/false
- impute_strategy: mean/median/most_frequent/none
- scale: standard/minmax/robust/none

Rules:
- missing=0 → impute=False
- skew > 1 → median
- normal → mean
- scaling optional

Columns:
{summary.to_string()}
"""

    # -------------------------
    # LLM CALL
    # -------------------------
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NumericalPreprocessingPlan,
            ),
        )
        # plan=llm.with_structured_output(NumericalPreprocessingPlan).invoke(prompt)

        plan = response.parsed

    except Exception as exc:
        return {
            "error": f"Numerical LLM failed: {exc}"
        }

    # -------------------------
    # BUILD PIPELINE
    # -------------------------
    transformers = []

    for col in plan.columns:

        steps = []

        # -------- Imputer --------
        if col.impute_strategy != "none":

            steps.append(
                (
                    "imputer",
                    SimpleImputer(
                        strategy=col.impute_strategy
                    )
                )
            )

        # -------- Scaler --------
        scaler = None

        if col.scale == "standard":
            scaler = StandardScaler()

        elif col.scale == "minmax":
            scaler = MinMaxScaler()

        elif col.scale == "robust":
            scaler = RobustScaler()

        if scaler:
            steps.append(("scaler", scaler))

        # -------- Column pipeline --------
        if steps:

            transformers.append(
                (
                    f"{col.name}_pipe",
                    Pipeline(steps),
                    [col.name]
                )
            )

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # -------------------------
    # SAVE PIPELINE
    # -------------------------
    os.makedirs("artifacts", exist_ok=True)

    save_path = "artifacts/numerical_pipeline.pkl"

    joblib.dump(pipeline, save_path)

    # -------------------------
    # RETURN STATE
    # -------------------------
    return {
        
        "numerical_pipeline_path": save_path,
        "numerical_plan": plan,
        
    }