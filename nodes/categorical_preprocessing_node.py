from typing import Literal
from pydantic import BaseModel
from sklearn.impute import SimpleImputer

from typing import Literal
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)
from llm_config import build_fallback_llm
from google import genai
from google.genai import types

class CategoricalColumnPlan(BaseModel):

    name: str

    impute_strategy: Literal[
        "most_frequent",
        "none"
    ]

    encoding: Literal[
        "onehot",
        "ordinal",
        "none"
    ]


class CategoricalPreprocessingPlan(BaseModel):

    columns: list[
        CategoricalColumnPlan
    ]



def categorical_preprocessing_node(state):
   

    train = pd.read_csv(state["train_path"])
    cols = state["categorical_columns"]

    # -------------------------
    # Summary stats
    # -------------------------
    summary = {}

    for c in cols:
        summary[c] = {
            "missing": float(train[c].isna().mean()),
            "unique": int(train[c].nunique())
        }

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

    client = genai.Client(api_key=api_key)

    prompt = f"""
Decide preprocessing for categorical columns.

Return structured JSON.

Rules:
- impute: true/false
- impute_strategy: most_frequent/none
- encoding: onehot/ordinal/none

Columns:
{summary}
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
                response_schema=CategoricalPreprocessingPlan,
            ),
        )
        # llm=build_fallback_llm()
        # plan=llm.with_structured_output(CategoricalPreprocessingPlan).invoke(prompt)
        plan = response.parsed

    except Exception as exc:
        return {
            "error": f"Categorical LLM failed: {exc}"
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

        # -------- Encoder --------
        encoder = None

        if col.encoding == "onehot":

            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        elif col.encoding == "ordinal":

            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

        if encoder:
            steps.append(("encoder", encoder))

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

    save_path = "artifacts/categorical_pipeline.pkl"

    joblib.dump(pipeline, save_path)

    # -------------------------
    # RETURN STATE
    # -------------------------
    return {
        
        "categorical_pipeline_path": save_path,
        "categorical_plan": plan
        
    }