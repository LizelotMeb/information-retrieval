"""One-off significance test: BM25+TAS-B vs BM25+RM3+TAS-B on per-query nDCG@10."""

from pathlib import Path

import pandas as pd
import pyterrier as pt
from scipy.stats import ttest_rel, wilcoxon

from src.pipelines import (
    evaluate_runs,
    init_pyterrier,
    sanitize_topics,
)

CACHE_DIR = Path("notebooks/results/cache")
FB_TERMS = 10
RERANK_K = 100


def load_cache(name: str) -> pd.DataFrame:
    return pd.read_pickle(CACHE_DIR / name)


def perquery_ndcg(runs: dict[str, pd.DataFrame], topics, qrels) -> pd.DataFrame:
    perq = evaluate_runs(
        runs, topics, qrels, eval_metrics=["ndcg_cut_10"], perquery=True
    )
    if {"measure", "value"}.issubset(perq.columns):
        perq = perq[perq["measure"] == "ndcg_cut_10"]
        wide = perq.pivot(index="qid", columns="name", values="value")
    else:
        wide = perq.pivot(index="qid", columns="name", values="ndcg_cut_10")
    return wide.sort_index()


def test_pair(wide: pd.DataFrame, label: str) -> None:
    base = wide["BM25+TAS-B"].to_numpy()
    comb = wide["BM25+RM3+TAS-B"].to_numpy()
    diff = comb - base

    n = len(diff)
    mean_base = base.mean()
    mean_comb = comb.mean()
    mean_diff = diff.mean()

    t_stat, t_p = ttest_rel(comb, base)
    # zero_method="wilcox" drops ties (default); alternative="two-sided"
    w_stat, w_p = wilcoxon(comb, base, zero_method="wilcox", alternative="two-sided")

    n_helped = int((diff > 0).sum())
    n_hurt = int((diff < 0).sum())
    n_tied = int((diff == 0).sum())

    print(f"\n=== {label} (n={n}) ===")
    print(f"  mean nDCG@10  BM25+TAS-B      = {mean_base:.4f}")
    print(f"  mean nDCG@10  BM25+RM3+TAS-B  = {mean_comb:.4f}")
    print(f"  mean delta (RM3+TAS-B − TAS-B)= {mean_diff:+.4f}")
    print(f"  queries helped / hurt / tied  = {n_helped} / {n_hurt} / {n_tied}")
    print(f"  paired t-test   t={t_stat:+.4f}  p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank W={w_stat:.4f}  p={w_p:.4f}")


def main() -> None:
    init_pyterrier()

    # --- Topics / qrels (same loading logic as experiments.ipynb) ---
    dl19_topics_ds = pt.get_dataset("irds:msmarco-passage/trec-dl-2019")
    dl19_qrels_ds = pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")
    dl20 = pt.get_dataset("irds:msmarco-passage/trec-dl-2020/judged")

    topics_19 = sanitize_topics(dl19_topics_ds.get_topics())
    qrels_19 = dl19_qrels_ds.get_qrels()
    topics_19 = topics_19[
        topics_19["qid"].astype(str).isin(qrels_19["qid"].astype(str))
    ].copy()

    topics_20 = sanitize_topics(dl20.get_topics())
    qrels_20 = dl20.get_qrels()

    # --- Cached runs (full configuration) ---
    runs_19 = {
        "BM25+TAS-B": load_cache(f"full_bm25_tasb_k{RERANK_K}_dl19.pkl"),
        "BM25+RM3+TAS-B": load_cache(
            f"full_bm25_rm3_fb{FB_TERMS}_tasb_k{RERANK_K}_dl19.pkl"
        ),
    }
    runs_20 = {
        "BM25+TAS-B": load_cache(f"full_bm25_tasb_k{RERANK_K}_dl20.pkl"),
        "BM25+RM3+TAS-B": load_cache(
            f"full_bm25_rm3_fb{FB_TERMS}_tasb_k{RERANK_K}_dl20.pkl"
        ),
    }

    wide_19 = perquery_ndcg(runs_19, topics_19, qrels_19)
    wide_20 = perquery_ndcg(runs_20, topics_20, qrels_20)

    test_pair(wide_19, "TREC DL 2019")
    test_pair(wide_20, "TREC DL 2020")


if __name__ == "__main__":
    main()
