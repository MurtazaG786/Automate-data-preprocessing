import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from llm_config import build_fallback_llm
from pydantic import BaseModel, ConfigDict, Field, model_validator

load_dotenv()

class ColumnClassification(BaseModel):
    model_config = ConfigDict(strict=True)

    numerical_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Column names where values represent measurable quantities suitable for arithmetic "
            "(e.g., age, salary, distance, score). Includes continuous and discrete counts "
            "with high cardinality (typically >10 unique values)."
        )
    )
    categorical_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Column names representing labels, classes, groups, binary flags, or ordinal categories — "
            "even if stored as integers. Typically <=10 unique values, but semantic meaning "
            "takes priority over cardinality (e.g., zip codes, IDs, yes/no flags)."
        )
    )

    @model_validator(mode="after")
    def no_overlap_and_no_missing(self) -> "ColumnClassification":
        overlap = set(self.numerical_columns) & set(self.categorical_columns)
        if overlap:
            raise ValueError(f"Columns appear in both lists: {overlap}")
        return self


def split_and_classify_node(state):
    """
    1. Reads the cleaned dataset
    2. Separates features (X) and target (y) if a target column exists
    3. Splits into train / test (80-20, stratified for classification)
    4. Sends the TRAINING columns + sample to the LLM for cat/num classification
    5. Saves train.csv and test.csv, returns column classification in state
    """

    output_path = state.get("output_file_path")
    target_column = state.get("target_column")
    problem_type = state.get("problem_type", "unsupervised")

    output_path = state.get("output_file_path")

    
    if not output_path or not os.path.exists(output_path):
        return {"error": "Processed dataset file not found for splitting."}  # ← silent fail here!
    ...

    df = pd.read_csv(output_path)

    # ── Train / Test Split ──────────────────────────────────────────────

    from sklearn.model_selection import train_test_split

    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]


        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        train_df = X_train.copy()
        train_df[target_column] = y_train

        test_df = X_test.copy()
        test_df[target_column] = y_test

        feature_columns = X_train.columns.tolist()
        sample_df = X_train.head(20)
    else:
        # No target column – unsupervised
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
        )

        feature_columns = df.columns.tolist()
        sample_df = train_df.head(20)


    base_dir = os.path.dirname(output_path)
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # ── LLM Column Classification ──────────────────────────────────────

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not api_key or not model_name:
        return {"error": "Missing GOOGLE_API_KEY or MODEL_NAME in environment."}

    dtypes_info = sample_df.dtypes.to_string()
    nunique_info = sample_df.nunique().to_string()

    prompt = f"""
You are a data scientist. Classify each feature column as "numerical" or "categorical".

Numerical: measurable quantities used for arithmetic (age, salary, score) — usually high cardinality (>10 unique values).
Categorical: labels, groups, flags, or IDs — even if stored as integers. Semantic meaning overrides cardinality (e.g., zip codes, binary flags).

Rules (apply in order):
1. Name/values suggest a label, group, or flag → categorical
2. Unique values <= 10 AND not a measurement → categorical
3. Arithmetic makes sense (averages, sums) → numerical
4. Uncertain → categorical

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
        # llm=build_fallback_llm()
        # classification=llm.with_structured_output(ColumnClassification).invoke(prompt)

        classification = ColumnClassification.model_validate_json(response.text)

        target = state.get(
            "target_column"
        )

        num_cols = [ c for c in classification.numerical_columns
            if (
                c in feature_columns
                and c != target
            )
        ]


        cat_cols = [
            c   for c in classification.categorical_columns
            if (
                c in feature_columns
                and c != target
            )
        ]

    except Exception as exc:
        return {"error": f"Column classification failed: {exc}"}
 

    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "steps": state.get("steps", []) + [
            f"Train/Test split done — Train: {len(train_df)} rows, Test: {len(test_df)} rows",
            f"Numerical columns ({len(num_cols)}): {num_cols}",
            f"Categorical columns ({len(cat_cols)}): {cat_cols}",
        ],
        "message": f"Split & classify complete. Num cols: {len(num_cols)}, Cat cols: {len(cat_cols)}",
        "error": None,
    }
