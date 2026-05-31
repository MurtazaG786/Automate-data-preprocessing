import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from nodes.llm_env import get_primary_api_key_model

load_dotenv()


class ColumnClassification(BaseModel):
    numerical_columns: list[str] = Field(
        description="List of numerical column names (int, float, continuous, discrete counts)"
    )
    categorical_columns: list[str] = Field(
        description="List of categorical column names (labels, classes, binary flags, ordinal text)"
    )


def classify_columns_node(state: dict[str, Any]) -> dict[str, Any]:
    train_path = state.get("train_path")
    target_column = state.get("target_column")

    if not train_path or not os.path.exists(train_path):
        return {"error": "Train dataset file not found for column classification."}

    train_df = pd.read_csv(train_path)
    feature_columns = [c for c in train_df.columns if c != target_column]

    if not feature_columns:
        return {
            "numerical_columns": [],
            "categorical_columns": [],
            "steps": state.get("steps", []) + [
                "No feature columns available for classification after feature engineering."
            ],
            "message": "No feature columns to classify.",
            "error": None,
        }

    api_key, model_name = get_primary_api_key_model()

    if not api_key or not model_name:
        return {"error": "Missing GOOGLE_API_KEY or MODEL_NAME in environment."}

    sample_df = train_df[feature_columns].head(20)
    dtypes_info = sample_df.dtypes.to_string()
    nunique_info = sample_df.nunique().to_string()

    prompt = f"""
You are a senior machine learning engineer.

Classify each feature column as either **numerical** or **categorical**.

Rules:
- Numerical columns contain continuous or discrete numbers used for computation (e.g., age, salary, price, quantity, score).
- Categorical columns contain labels, classes, or binary flags, EVEN if they are stored as integers (e.g., gender encoded as 0/1, department IDs with few unique values).
- A column stored as int/float but having very few unique values (e.g., <= 10) is likely categorical.
- Only classify the feature columns listed below. Do NOT include the target column.
- Return ONLY valid column names from the given list.

Feature columns:
{feature_columns}

Column dtypes:
{dtypes_info}

Unique value counts:
{nunique_info}

Sample data (from training set only):
{sample_df.to_string()}
"""

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ColumnClassification,
            ),
        )

        classification = ColumnClassification.model_validate_json(response.text)

        num_cols = [
            c for c in classification.numerical_columns
            if c in feature_columns and c != target_column
        ]

        cat_cols = [
            c for c in classification.categorical_columns
            if c in feature_columns and c != target_column
        ]

    except Exception as exc:
        return {"error": f"Column classification failed: {exc}"}

    if not num_cols and not cat_cols:
        # Heuristic fallback when LLM returns nothing
        num_cols = []
        cat_cols = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                # Low cardinality numeric can be categorical
                if train_df[col].nunique(dropna=True) <= 10:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
            else:
                cat_cols.append(col)

    return {
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "steps": state.get("steps", []) + [
            f"Column classification updated after feature engineering. Num: {len(num_cols)}, Cat: {len(cat_cols)}",
        ],
        "message": "Column classification updated.",
        "error": None,
    }
