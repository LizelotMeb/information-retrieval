from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyterrier as pt
from pyterrier_dr import TasB


DEFAULT_METRICS = ["ndcg_cut_10", "map", "recall_100"]


def init_pyterrier(mem: str | None = None) -> None:
    """Start the PyTerrier Java backend once."""
    if not pt.java.started():
        kwargs = {"mem": mem} if mem else {}
        pt.java.init(**kwargs)


def subset_topics(
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
    n_topics: int | None = None,
    qids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a smaller topic/qrels subset for fast iterations."""
    if qids is not None:
        qid_set = {str(qid) for qid in qids}
        selected_topics = topics[topics["qid"].astype(str).isin(qid_set)].copy()
    elif n_topics is not None:
        selected_topics = topics.head(n_topics).copy()
    else:
        selected_topics = topics.copy()

    selected_qids = selected_topics["qid"].astype(str)
    selected_qrels = qrels[qrels["qid"].astype(str).isin(selected_qids)].copy()
    return selected_topics, selected_qrels


def sanitize_topics(topics: pd.DataFrame) -> pd.DataFrame:
    """Normalize topic text so Terrier does not parse punctuation as query syntax."""
    output = topics.copy()
    output["query"] = (
        output["query"]
        .astype(str)
        .str.replace(":", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return output


def build_bm25(index, num_results: int = 1000):
    return pt.terrier.Retriever(index, wmodel="BM25", num_results=num_results)


def build_bm25_rm3(index, num_results: int = 1000, fb_terms: int = 10):
    bm25 = build_bm25(index, num_results=num_results)
    rm3 = pt.rewrite.RM3(index, fb_terms=fb_terms)
    return bm25 >> rm3 >> bm25


def build_bm25_tasb(
    index,
    irds_dataset,
    num_results: int = 1000,
    rerank_k: int = 100,
):
    bm25 = build_bm25(index, num_results=num_results)
    get_text = pt.text.get_text(irds_dataset, "text")
    tasb = TasB.dot()
    return bm25 % rerank_k >> get_text >> tasb


def build_bm25_rm3_tasb(
    index,
    irds_dataset,
    num_results: int = 1000,
    fb_terms: int = 10,
    rerank_k: int = 100,
):
    bm25 = build_bm25(index, num_results=num_results)
    rm3 = pt.rewrite.RM3(index, fb_terms=fb_terms)
    get_text = pt.text.get_text(irds_dataset, "text")
    tasb = TasB.dot()

    # TAS-B should score the original natural-language query, not the RM3 rewrite.
    reset_query = pt.apply.generic(lambda df: df.assign(query=df["query_0"]))
    return bm25 >> rm3 >> bm25 % rerank_k >> reset_query >> get_text >> tasb


def run_and_cache(
    pipeline,
    topics: pd.DataFrame,
    cache_path: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run a pipeline once and reuse the saved resultset on later runs."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not overwrite:
        return pd.read_pickle(cache_path)

    results = pipeline.transform(topics)
    results.to_pickle(cache_path)
    return results


def evaluate_runs(
    runs: dict[str, pd.DataFrame],
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
    eval_metrics: list[str] | None = None,
    perquery: bool = False,
) -> pd.DataFrame:
    """Evaluate cached resultsets without rerunning the underlying pipelines."""
    names = list(runs)
    systems = [runs[name] for name in names]
    metrics = eval_metrics or DEFAULT_METRICS
    return pt.Experiment(
        systems,
        topics,
        qrels,
        eval_metrics=metrics,
        names=names,
        perquery=perquery,
    )


def perquery_comparison(
    runs: dict[str, pd.DataFrame],
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
    metric: str = "ndcg_cut_10",
) -> pd.DataFrame:
    """Return one row per qid so systems are easy to compare query by query."""
    perquery = evaluate_runs(
        runs,
        topics,
        qrels,
        eval_metrics=[metric],
        perquery=True,
    )

    if {"qid", "name", metric}.issubset(perquery.columns):
        wide = perquery.pivot(index="qid", columns="name", values=metric)
    elif {"qid", "name", "measure", "value"}.issubset(perquery.columns):
        metric_rows = perquery[perquery["measure"] == metric]
        wide = metric_rows.pivot(index="qid", columns="name", values="value")
    else:
        raise ValueError(
            "Unexpected per-query output format from pt.Experiment; "
            f"got columns: {sorted(perquery.columns)}"
        )

    wide = wide.reset_index()
    topic_queries = topics[["qid", "query"]].drop_duplicates()
    wide["qid"] = wide["qid"].astype(str)
    topic_queries["qid"] = topic_queries["qid"].astype(str)
    return topic_queries.merge(wide, on="qid", how="left")


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
