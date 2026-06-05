from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


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
    denom = df[denom_col].replace(0, np.nan)
    return df[num_col] / denom


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


def _filter_inputs(inputs: list[str], columns: list[str], target_column: str | None) -> list[str]:
    return [c for c in inputs if c in columns and c != target_column]


def _compute_bins(series: pd.Series, op: dict[str, Any]) -> list[float]:
    method = op.get("method") or "quantile"
    if method == "fixed" and op.get("bins"):
        return [float(v) for v in op["bins"]]
    q_val = int(op.get("q") or 4)
    _, bins_out = pd.qcut(series, q=q_val, duplicates="drop", retbins=True)
    return bins_out.tolist()


def _compute_allowed(series: pd.Series, op: dict[str, Any]) -> set[str]:
    counts = series.astype(str).value_counts()
    min_pct = op.get("min_pct")
    min_freq = op.get("min_freq")
    if min_pct is not None:
        return set(counts[counts / counts.sum() >= float(min_pct)].index)
    if min_freq is not None:
        return set(counts[counts >= int(min_freq)].index)
    return set(counts[counts >= 10].index)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, plan: dict[str, Any] | None = None, target_column: str | None = None):
        self.plan = plan or {"ops": []}
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineeringTransformer":
        df = pd.DataFrame(X).copy()
        self._ops = list(self.plan.get("ops") or [])
        self._bin_edges: dict[str, list[float]] = {}
        self._rare_allowed: dict[str, set[str]] = {}
        self._drop_cols: set[str] = set()

        feature_cols = [c for c in df.columns if c != self.target_column]

        for op in self._ops:
            op_type = op.get("op")
            inputs = _filter_inputs(op.get("inputs", []), df.columns.tolist(), self.target_column)
            output_column = op.get("output_column")

            if op_type == "bin_numeric" and inputs and output_column:
                self._bin_edges[output_column] = _compute_bins(df[inputs[0]], op)

            elif op_type == "rare_to_other" and inputs and output_column:
                self._rare_allowed[output_column] = _compute_allowed(df[inputs[0]], op)

        transformed = self._apply_ops(df)

        for op in self._ops:
            op_type = op.get("op")
            inputs = _filter_inputs(op.get("inputs", []), transformed.columns.tolist(), self.target_column)

            if op_type == "drop_low_variance":
                drop_cols = _drop_low_variance(
                    transformed,
                    inputs or [c for c in transformed.columns if c != self.target_column],
                    op.get("threshold")
                )
                self._drop_cols.update(drop_cols)

            elif op_type == "drop_low_correlation" and y is not None:
                if pd.api.types.is_numeric_dtype(pd.Series(y)):
                    corr_threshold = float(op.get("corr_threshold") or 0.02)
                    candidate_cols = inputs or [
                        c for c in transformed.columns if c != self.target_column
                    ]
                    numeric_cols = [
                        c for c in candidate_cols
                        if c in transformed.columns and pd.api.types.is_numeric_dtype(transformed[c])
                    ]
                    if numeric_cols:
                        corr = transformed[numeric_cols].corrwith(pd.Series(y)).abs()
                        drop_cols = corr[corr < corr_threshold].index.tolist()
                        self._drop_cols.update(drop_cols)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        transformed = self._apply_ops(df)

        if self._drop_cols:
            transformed = transformed.drop(
                columns=[c for c in self._drop_cols if c in transformed.columns]
            )

        return transformed

    def _apply_ops(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()

        for op in self._ops:
            op_type = op.get("op")
            inputs = _filter_inputs(op.get("inputs", []), transformed.columns.tolist(), self.target_column)
            output_column = op.get("output_column")

            if op_type == "extract_numeric" and output_column and inputs:
                transformed[output_column] = _extract_numeric(transformed[inputs[0]])

            elif op_type == "concat_text" and output_column and inputs:
                sep = op.get("sep") or " "
                transformed[output_column] = _concat_text(transformed, inputs, sep)

            elif op_type == "date_parts" and output_column and inputs:
                parts = op.get("parts") or ["year", "month", "day"]
                part_map = _date_parts(transformed[inputs[0]], parts)
                for part_name, series in part_map.items():
                    col_name = f"{output_column}_{part_name}"
                    transformed[col_name] = series

            elif op_type == "ratio" and output_column and len(inputs) >= 2:
                transformed[output_column] = _ratio(transformed, inputs[0], inputs[1])

            elif op_type == "bin_numeric" and output_column and inputs:
                bins = self._bin_edges.get(output_column) or _compute_bins(transformed[inputs[0]], op)
                transformed[output_column] = pd.cut(
                    transformed[inputs[0]],
                    bins=bins,
                    include_lowest=True,
                ).astype(str)

            elif op_type == "rare_to_other" and output_column and inputs:
                allowed = self._rare_allowed.get(output_column) or _compute_allowed(transformed[inputs[0]], op)
                transformed[output_column] = transformed[inputs[0]].astype(str).apply(
                    lambda v: v if str(v) in allowed else "Other"
                )

        return transformed.replace({pd.NA: np.nan})
