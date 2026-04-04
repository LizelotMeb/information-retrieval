from __future__ import annotations

from typing import Iterable

import pandas as pd


def add_gain_columns(
    comparison: pd.DataFrame,
    baseline: str,
    systems: Iterable[str],
    suffix: str = "_gain",
) -> pd.DataFrame:
    """Add score deltas versus a baseline system to a per-query table."""
    output = comparison.copy()
    for system in systems:
        output[f"{system}{suffix}"] = output[system] - output[baseline]
    return output


def add_query_features(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add simple query-level features for lightweight error analysis."""
    output = comparison.copy()
    query_text = output["query"].fillna("").astype(str)

    output["query_len_words"] = query_text.str.split().str.len()
    output["query_len_chars"] = query_text.str.len()
    output["has_colon"] = query_text.str.contains(":", regex=False)
    output["has_digit"] = query_text.str.contains(r"\d", regex=True)
    output["is_question"] = query_text.str.endswith("?")
    return output


def label_gain(
    comparison: pd.DataFrame,
    gain_column: str,
    positive_threshold: float = 0.01,
    negative_threshold: float = -0.01,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Label each query as helped, hurt, or neutral."""
    output = comparison.copy()
    label_column = label_column or f"{gain_column}_label"

    def classify(value: float) -> str:
        if value >= positive_threshold:
            return "helped"
        if value <= negative_threshold:
            return "hurt"
        return "neutral"

    output[label_column] = output[gain_column].apply(classify)
    return output


def top_queries(
    comparison: pd.DataFrame,
    sort_column: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the most improved or most harmed queries for a chosen metric."""
    columns = ["qid", "query", sort_column]
    extra_columns = [
        column
        for column in comparison.columns
        if column not in columns and not column.endswith("_label")
    ]
    selected = [column for column in columns + extra_columns if column in comparison.columns]
    return comparison.sort_values(sort_column, ascending=ascending)[selected].head(n)


def summarize_labels(
    comparison: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Count how many queries were helped, hurt, or unchanged."""
    summary = comparison[label_column].value_counts(dropna=False).rename_axis(label_column)
    return summary.reset_index(name="count")


def compare_query_sets(
    comparison: pd.DataFrame,
    label_column: str,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Summarize average feature values for helped, hurt, and neutral queries."""
    selected = [label_column] + [column for column in feature_columns if column in comparison.columns]
    return comparison[selected].groupby(label_column, dropna=False).mean(numeric_only=True).reset_index()
