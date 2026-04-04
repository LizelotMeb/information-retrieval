# Information Retrieval Project

This project investigates the interaction between classical query expansion and neural re-ranking methods in Information Retrieval (IR).

## Requirements

Before running the notebook, make sure you have:

- **Python 3.14**
- **Java 11 or newer** (`python-terrier` requires Java)

If Java is installed but not detected, set `JAVA_HOME` to your Java installation path.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Start Jupyter Notebook from the project root:

```bash
jupyter notebook
```

Then open `notebooks/experiments.ipynb`.

## How The Code Is Organized

The project is split into two parts:

- `src/pipelines.py`: reusable helper code for building retrieval pipelines, caching ranking outputs, and comparing systems per query
- `notebooks/experiments.ipynb`: the main notebook that imports those helpers and runs the experiments

`src/pipelines.py` is used for:

- building `BM25`, `BM25 + RM3`, `BM25 + TAS-B`, and `BM25 + RM3 + TAS-B`
- caching pipeline outputs in `results/cache/` so expensive runs do not have to be repeated
- evaluating saved runs without recomputing them
- making per-query comparison tables

`notebooks/experiments.ipynb` is used for:

- loading the datasets and Terrier index
- choosing a fast or full configuration
- calling the shared pipeline helpers
- analyzing aggregate and per-query results

## Recommended Workflow

The notebook is designed so you do not need to rerun everything every time.

1. Run the setup and data-loading cells
2. Run the cache-building cells once so ranking outputs are written to `results/cache/`
3. Re-run only the evaluation and analysis cells when you want to compare systems or inspect specific queries

Use `FAST_MODE = True` while developing. This runs fewer topics and uses a smaller reranking depth. Switch to `FAST_MODE = False` only for the final full experiment run.

The `results/cache/` directory is created locally by the notebook and is not tracked in Git. Each user should run the notebook once to generate their own cached result files.

## Research Goal

Both pseudo-relevance feedback (PRF) and neural re-ranking are commonly used to address the vocabulary mismatch problem. While both methods are effective individually, it is unclear how they interact and under which conditions query expansion helps or harms neural ranking.

This project studies:
- Whether pseudo-relevance feedback improves neural re-ranking performance
- How the number of expansion terms (fb_terms) influences this interaction

---

## Methods

We implement and compare the following retrieval pipelines using PyTerrier:

1. **BM25** (baseline lexical retrieval)
2. **BM25 + RM3** (query expansion)
3. **BM25 + TAS-B** (neural re-ranking)
4. **BM25 + RM3 + TAS-B** (combined pipeline)

To analyze the effect of query expansion, we vary the RM3 parameter `fb_terms`.

---

## Dataset

We use the **MS MARCO Passage** collection together with the **TREC Deep Learning 2019** and **2020 judged** topic sets:

- `msmarco_passage` for the pre-built Terrier index
- `irds:msmarco-passage` for document text lookup
- `irds:msmarco-passage/trec-dl-2019/judged` for DL19 topics and qrels
- `irds:msmarco-passage/trec-dl-2020/judged` for DL20 topics and qrels

---

## Repository Structure

```text
notebooks/
    experiments.ipynb    # main notebook for running and analyzing experiments

results/
    cache/               # cached ranking outputs for reuse across runs

src/
    pipelines.py         # reusable pipeline, caching, and per-query helpers

requirements.txt         # dependencies
README.md
```
