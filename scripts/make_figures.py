"""Generate report figures from cached per-query runs.

Figure 1: fb_terms sweep for BM25+RM3+TAS-B on NDCG@10, MAP, Recall@100.
Figure 2: per-query NDCG@10 gain histogram (BM25+RM3+TAS-B - BM25+TAS-B).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyterrier as pt

from src.pipelines import evaluate_runs, init_pyterrier, sanitize_topics

CACHE_DIR = Path("notebooks/results/cache")
FIG_DIR = Path("results/figures")

FB_VALUES = [5, 10, 20, 30, 50]
RERANK_K = 100
METRICS = ["ndcg_cut_10", "map", "recall_100"]
METRIC_LABELS = {
    "ndcg_cut_10": "NDCG@10",
    "map": "MAP",
    "recall_100": "Recall@100",
}


def load(name: str) -> pd.DataFrame:
    return pd.read_pickle(CACHE_DIR / name)


def load_topics_qrels():
    init_pyterrier()
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
    return (topics_19, qrels_19), (topics_20, qrels_20)


def sweep_scores(topics, qrels, dl_tag: str) -> pd.DataFrame:
    runs = {"BM25+TAS-B": load(f"full_bm25_tasb_k{RERANK_K}_dl{dl_tag}.pkl")}
    for fb in FB_VALUES:
        runs[f"fb={fb}"] = load(
            f"full_bm25_rm3_fb{fb}_tasb_k{RERANK_K}_dl{dl_tag}.pkl"
        )
    df = evaluate_runs(runs, topics, qrels, eval_metrics=METRICS)
    return df.set_index("name")


def figure1_fb_sweep(scores_19, scores_20, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, metric in zip(axes, METRICS):
        base_19 = scores_19.loc["BM25+TAS-B", metric]
        base_20 = scores_20.loc["BM25+TAS-B", metric]
        vals_19 = [scores_19.loc[f"fb={fb}", metric] for fb in FB_VALUES]
        vals_20 = [scores_20.loc[f"fb={fb}", metric] for fb in FB_VALUES]

        ax.plot(FB_VALUES, vals_19, marker="o", color="C0", label="DL2019")
        ax.plot(FB_VALUES, vals_20, marker="s", color="C1", label="DL2020")
        ax.axhline(base_19, linestyle="--", alpha=0.45, color="C0", linewidth=1)
        ax.axhline(base_20, linestyle="--", alpha=0.45, color="C1", linewidth=1)
        ax.set_xlabel("fb_terms", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_xticks(FB_VALUES)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def perquery_delta(topics, qrels, dl_tag: str) -> np.ndarray:
    runs = {
        "BM25+TAS-B": load(f"full_bm25_tasb_k{RERANK_K}_dl{dl_tag}.pkl"),
        "BM25+RM3+TAS-B": load(
            f"full_bm25_rm3_fb10_tasb_k{RERANK_K}_dl{dl_tag}.pkl"
        ),
    }
    perq = evaluate_runs(
        runs, topics, qrels, eval_metrics=["ndcg_cut_10"], perquery=True
    )
    if {"measure", "value"}.issubset(perq.columns):
        perq = perq[perq["measure"] == "ndcg_cut_10"]
        wide = perq.pivot(index="qid", columns="name", values="value")
    else:
        wide = perq.pivot(index="qid", columns="name", values="ndcg_cut_10")
    return (wide["BM25+RM3+TAS-B"] - wide["BM25+TAS-B"]).to_numpy()


def figure2_per_query_hist(
    topics_19, qrels_19, topics_20, qrels_20, out_path: Path
) -> None:
    d19 = perquery_delta(topics_19, qrels_19, "19")
    d20 = perquery_delta(topics_20, qrels_20, "20")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    bins = np.linspace(-0.5, 0.5, 21)
    for ax, data, title in zip(axes, [d19, d20], ["DL2019", "DL2020"]):
        ax.hist(data, bins=bins, edgecolor="black", alpha=0.85, color="C0")
        ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlabel(r"$\Delta$ NDCG@10 (RM3+TAS-B $-$ TAS-B)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("# queries", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (topics_19, qrels_19), (topics_20, qrels_20) = load_topics_qrels()

    scores_19 = sweep_scores(topics_19, qrels_19, "19")
    scores_20 = sweep_scores(topics_20, qrels_20, "20")
    figure1_fb_sweep(scores_19, scores_20, FIG_DIR / "fb_terms_sweep.pdf")
    print("wrote", FIG_DIR / "fb_terms_sweep.pdf")
    print("wrote", FIG_DIR / "fb_terms_sweep.png")

    figure2_per_query_hist(
        topics_19, qrels_19, topics_20, qrels_20, FIG_DIR / "per_query_gain_hist.pdf"
    )
    print("wrote", FIG_DIR / "per_query_gain_hist.pdf")
    print("wrote", FIG_DIR / "per_query_gain_hist.png")


if __name__ == "__main__":
    main()
