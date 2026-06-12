import json
import os
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from nodes.llm_env import get_primary_api_key_model
import shutil
from pathlib import Path
import os, zipfile

load_dotenv()


class FeatureOp(BaseModel):
    op: Literal[
        "extract_numeric",
        "concat_text",
        "date_parts",
        "ratio",
        "bin_numeric",
        "rare_to_other",
        "drop_low_variance",
        "drop_low_correlation",
    ]
    output_column: str | None = Field(
        description="Name of the new column to create. For drop_low_variance this can be null."
    )
    inputs: list[str] = Field(
        description="Input columns used by the operation."
    )
    sep: str | None = Field(default=None, description="Separator for concat_text")
    parts: list[str] | None = Field(default=None, description="Date parts for date_parts")
    method: Literal["quantile", "fixed"] | None = Field(default=None, description="Binning method")
    bins: list[float] | None = Field(default=None, description="Fixed bins for bin_numeric")
    q: int | None = Field(default=None, description="Quantiles for bin_numeric")
    min_freq: int | None = Field(default=None, description="Minimum frequency for rare_to_other")
    min_pct: float | None = Field(default=None, description="Minimum percentage for rare_to_other")
    threshold: float | None = Field(default=None, description="Variance threshold for drop_low_variance")
    corr_threshold: float | None = Field(default=None, description="Correlation threshold for drop_low_correlation")
    reason: str = Field(description="Short reason for this transformation.")
    confidence: float = Field(description="Confidence score between 0 and 1.")


class FeatureEngineeringPlan(BaseModel):
    ops: list[FeatureOp]
    notes: str | None = None


def _extract_numeric(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _concat_text(df: pd.DataFrame, inputs: list[str], sep: str) -> pd.Series:
    parts = [df[col].fillna("").astype(str) for col in inputs]
    return parts[0].str.cat(parts[1:], sep=sep) if parts else pd.Series(dtype=str)


def _date_parts(series: pd.Series, parts: list[str]) -> dict[str, pd.Series]:
    dt = pd.to_datetime(series, errors="coerce")
    outputs: dict[str, pd.Series] = {}
    if "year" in parts:
        outputs["year"] = dt.dt.year
    if "month" in parts:
        outputs["month"] = dt.dt.month
    if "day" in parts:
        outputs["day"] = dt.dt.day
    return outputs


def _ratio(df: pd.DataFrame, num_col: str, denom_col: str) -> pd.Series:
    denom = df[denom_col].replace(0, pd.NA)
    return df[num_col] / denom


def _bin_numeric(series: pd.Series, method: str | None, bins: list[float] | None, q: int | None) -> tuple[pd.Series, list[float]]:
    chosen_method = method or "quantile"
    if chosen_method == "fixed" and bins:
        return pd.cut(series, bins=bins, include_lowest=True).astype(str), bins
    q_val = int(q or 4)
    binned, bins_out = pd.qcut(series, q=q_val, duplicates="drop", retbins=True)
    return binned.astype(str), bins_out.tolist()


def _rare_to_other(series: pd.Series, allowed: set[str]) -> pd.Series:
    return series.apply(lambda v: v if str(v) in allowed else "Other")


def _drop_low_variance(df: pd.DataFrame, cols: list[str], threshold: float | None) -> list[str]:
    threshold_val = float(threshold or 0.0)
    drop_cols: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].var(skipna=True) <= threshold_val:
                drop_cols.append(col)
        else:
            if df[col].nunique(dropna=True) <= 1:
                drop_cols.append(col)
    return drop_cols

def copy_feature_engineering_transformer_to_temp(temp_dir: str) -> str:
    """
    Copies the reusable sklearn transformer file into temp_dir/nodes/
    so it can be bundled with preprocessor.pkl.
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

def feature_engineering_node(state: dict[str, Any]) -> dict[str, Any]:
    train_path = state.get("train_path")
    test_path = state.get("test_path")
    target = state.get("target_column")
    temp_dir = state.get("temp_dir")

    if not train_path or not os.path.exists(train_path):
        return {"error": "Train dataset file not found for feature engineering."}

    if not test_path or not os.path.exists(test_path):
        return {"error": "Test dataset file not found for feature engineering."}

    if state.get("feature_engineering_applied") is True:
        return {
            "steps": state.get("steps", []) + ["Feature engineering already applied. Skipping."],
            "error": None,
        }

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature_cols = [c for c in train.columns if c != target]

    if not feature_cols:
        return {
            "steps": state.get("steps", []) + [
                "No feature columns available for feature engineering."
            ],
            "message": "Feature engineering skipped (no feature columns).",
            "error": None,
        }

    # If a plan already exists in state, request approval without regenerating.
    existing_plan = state.get("feature_engineering_plan")
    if existing_plan and state.get("feature_engineering_approved") is not True:
        decision = interrupt({
            "type": "feature_engineering_plan",
            "plan": existing_plan,
            "question": "Approve the feature engineering plan?",
        })
        if decision.get("approved") is not True:
            return {
                "feature_engineering_approved": False,
                "steps": state.get("steps", []) + ["Feature engineering rejected by user."],
                "message": "Feature engineering skipped by user.",
                "error": None,
            }
        plan = FeatureEngineeringPlan.model_validate(existing_plan)
        plan_dict = existing_plan
    elif existing_plan and state.get("feature_engineering_approved") is True:
        plan = FeatureEngineeringPlan.model_validate(existing_plan)
        plan_dict = existing_plan
    else:
        plan = None
        plan_dict = None

    if plan is None:
        api_key, model_name = get_primary_api_key_model()

        if not api_key or not model_name:
            return {"error": "Missing GOOGLE_API_KEY or MODEL_NAME in environment."}

        summary = train[feature_cols].describe(include="all").T
        missing = train[feature_cols].isna().mean().rename("missing_rate")
        nunique = train[feature_cols].nunique().rename("unique_count")
        target_info = "No target column"
        if target and target in train.columns:
            if pd.api.types.is_numeric_dtype(train[target]):
                target_info = train[target].describe().to_string()
            else:
                target_info = train[target].astype(str).value_counts().head(10).to_string()

        prompt = f"""
You are a senior ML feature engineering expert.

Propose a safe, minimal set of feature engineering operations.
Write explanations for a non-technical audience.

Rules:
- Only use these ops: extract_numeric, concat_text, date_parts, ratio, bin_numeric, rare_to_other, drop_low_variance, drop_low_correlation.
- Never touch the target column.
- Only use columns listed in feature_columns.
- Keep ops minimal and high-impact; return at most 4 ops.
- Only include fields relevant to the chosen op. Do not include nulls or unused fields.
- Always include: op, inputs, reason, confidence.
- The reason must be plain-English, non-technical, and short (one sentence). Avoid ML jargon.
- For ops that create a new feature, include output_column.
- For concat_text include sep.
- For date_parts include parts.
- For bin_numeric include method and (bins or q).
- For rare_to_other include min_freq or min_pct.
- For drop_low_variance include threshold (use 0.0 if unsure).
- For drop_low_correlation include corr_threshold (default 0.02) and only if target is numeric.
- Avoid redundant ops on the same input columns.

Return JSON matching the schema.

feature_columns:
{feature_cols}

summary:
{summary.to_string()}

missing_rate:
{missing.to_string()}

unique_count:
{nunique.to_string()}

sample_rows:
{train[feature_cols].head(20).to_string()}

target_info:
{target_info}
"""

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FeatureEngineeringPlan,
                ),
            )
            plan = response.parsed
        except Exception as exc:
            return {"error": f"Feature engineering plan generation failed: {exc}"}

        # Keep only high-confidence ops and cap total count
        filtered_ops = [op for op in plan.ops if op.confidence >= 0.6]
        plan.ops = filtered_ops[:6]
        plan_dict = plan.model_dump(exclude_none=True)

        # HITL approval
        decision = interrupt({
            "type": "feature_engineering_plan",
            "plan": plan_dict,
            "question": "Approve the feature engineering plan?",
        })

        if decision.get("approved") is not True:
            return {
                "feature_engineering_plan": plan_dict,
                "feature_engineering_approved": False,
                "steps": state.get("steps", []) + ["Feature engineering rejected by user."],
                "message": "Feature engineering skipped by user.",
                "error": None,
            }

    applied_ops: list[str] = []

    for op in plan.ops:
        inputs = [c for c in op.inputs if c in train.columns and c != target]
        if not inputs and op.op != "drop_low_variance":
            continue

        if op.op == "extract_numeric" and op.output_column:
            train[op.output_column] = _extract_numeric(train[inputs[0]])
            test[op.output_column] = _extract_numeric(test[inputs[0]])
            applied_ops.append(op.output_column)

        elif op.op == "concat_text" and op.output_column:
            sep = op.sep or " "
            train[op.output_column] = _concat_text(train, inputs, sep)
            test[op.output_column] = _concat_text(test, inputs, sep)
            applied_ops.append(op.output_column)

        elif op.op == "date_parts" and op.output_column:
            parts = op.parts or ["year", "month", "day"]
            part_map = _date_parts(train[inputs[0]], parts)
            for part_name, series in part_map.items():
                col_name = f"{op.output_column}_{part_name}"
                train[col_name] = series
                test[col_name] = _date_parts(test[inputs[0]], parts).get(part_name)
                applied_ops.append(col_name)

        elif op.op == "ratio" and op.output_column and len(inputs) >= 2:
            train[op.output_column] = _ratio(train, inputs[0], inputs[1])
            test[op.output_column] = _ratio(test, inputs[0], inputs[1])
            applied_ops.append(op.output_column)

        elif op.op == "bin_numeric" and op.output_column:
            train_binned, bins = _bin_numeric(train[inputs[0]], op.method, op.bins, op.q)
            train[op.output_column] = train_binned
            test[op.output_column] = pd.cut(
                test[inputs[0]],
                bins=bins,
                include_lowest=True
            ).astype(str)
            applied_ops.append(op.output_column)

        elif op.op == "rare_to_other" and op.output_column:
            min_freq = op.min_freq
            min_pct = op.min_pct
            counts = train[inputs[0]].astype(str).value_counts()
            if min_pct is not None:
                allowed = set(counts[counts / counts.sum() >= float(min_pct)].index)
            elif min_freq is not None:
                allowed = set(counts[counts >= int(min_freq)].index)
            else:
                allowed = set(counts[counts >= 10].index)
            train[op.output_column] = _rare_to_other(train[inputs[0]].astype(str), allowed)
            test[op.output_column] = _rare_to_other(test[inputs[0]].astype(str), allowed)
            applied_ops.append(op.output_column)

        elif op.op == "drop_low_variance":
            drop_cols = _drop_low_variance(train, inputs or feature_cols, op.threshold)
            if drop_cols:
                train.drop(columns=drop_cols, inplace=True)
                test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)
                applied_ops.extend(drop_cols)

        elif op.op == "drop_low_correlation":
            if target and target in train.columns and pd.api.types.is_numeric_dtype(train[target]):
                corr_threshold = float(op.corr_threshold or 0.02)
                numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
                if numeric_cols:
                    corr = train[numeric_cols].corrwith(train[target]).abs()
                    drop_cols = corr[corr < corr_threshold].index.tolist()
                    if drop_cols:
                        train.drop(columns=drop_cols, inplace=True)
                        test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)
                        applied_ops.extend(drop_cols)

    base_dir = temp_dir or os.path.dirname(train_path)
    train_out = os.path.join(base_dir, "train_fe.csv")
    test_out = os.path.join(base_dir, "test_fe.csv")

    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)

    plan_path = os.path.join(base_dir, "feature_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f, indent=2)

    feature_engineering_transformer_path = copy_feature_engineering_transformer_to_temp(base_dir)

    return {
    "train_raw_path": state.get("train_raw_path") or train_path,
    "test_raw_path": state.get("test_raw_path") or test_path,
    "train_path": train_out,
    "test_path": test_out,
    "feature_engineering_plan": plan_dict,
    "feature_engineering_plan_path": plan_path,
    "feature_engineering_transformer_path": feature_engineering_transformer_path,
    "feature_engineering_approved": True,
    "feature_engineering_applied": True,
    "steps": state.get("steps", []) + [
        f"Feature engineering applied. Ops affected: {len(applied_ops)}",
        "Feature engineering transformer file created.",
    ],
    "message": "Feature engineering applied.",
    "error": None,
}
