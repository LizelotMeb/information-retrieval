# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating the interaction between classical query expansion (RM3 pseudo-relevance feedback) and neural re-ranking (TAS-B) in information retrieval pipelines. Uses the MS MARCO Passage dataset with TREC Deep Learning 2019 and 2020 evaluation queries.

**Research question**: Does pseudo-relevance feedback improve neural re-ranking performance, and how does the amount of query expansion influence this interaction?

## Tech Stack

- **PyTerrier** — Python wrapper around the Java Terrier IR framework; provides pipeline composition, indexing, and evaluation
- **pyterrier_dr** — PyTerrier plugin for dense retrieval; provides the TAS-B bi-encoder model
- **Jupyter Notebook** — all experimentation lives in `notebooks/experiments.ipynb`
- **Pandas** — result manipulation and analysis
- Requires a Java runtime (PyTerrier starts a JVM internally)

## Dependencies

No `requirements.txt` is currently tracked. Key pip packages needed: `python-terrier`, `pyterrier_dr`, `pandas`, `jupyter`.

## Running Experiments

```bash
jupyter notebook notebooks/experiments.ipynb
```

The notebook uses a pre-built Terrier index for MS MARCO Passage downloaded via `dataset.get_index(variant="terrier_stemmed")`. First run will download the index and corpus (~several GB).

## Architecture

All pipeline logic is in `notebooks/experiments.ipynb`. `src/pipelines.py` exists as an empty placeholder.

**Four retrieval pipelines:**
1. **BM25** — baseline lexical retrieval (top 1000)
2. **BM25 + RM3** — query expansion via pseudo-relevance feedback
3. **BM25 + TAS-B** — BM25 top-100 re-ranked by TAS-B neural model
4. **BM25 + RM3 + TAS-B** — RM3 expands query for BM25, then TAS-B re-ranks top-100 using the **original** query (reset via `pt.apply.query`)

**Pipeline composition** uses PyTerrier's `>>` operator and `%` for rank cutoff:
```python
bm25 >> rm3 >> bm25 % 100 >> reset_query >> get_text >> tasb
```

**Key design decision**: After RM3 expansion, the query is reset to the original natural language before TAS-B scoring, because TAS-B is trained on natural language queries, not weighted term expansions.

**Evaluation** uses `pt.Experiment()` with metrics: `ndcg_cut_10`, `map`, `recall_100`. Results are reported on both DL 2019 and DL 2020.

**Parameter study**: RM3 `fb_terms` is varied across [5, 10, 20, 30, 50] to study its effect on both lexical retrieval and neural re-ranking.
