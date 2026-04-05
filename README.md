# Information Retrieval Project

This project investigates the interaction between classical query expansion and neural re-ranking methods in Information Retrieval (IR).

## Requirements

Before running the notebook, make sure you have:

- **Python 3.12**
- **Java 11 or newer** (`python-terrier` requires Java)


## Java Setup

`python-terrier` needs a working Java installation. A common failure case is that Java is installed, but `JAVA_HOME` is missing or points to the wrong location.

Check whether Java is available:

```bash
java -version
echo $JAVA_HOME
```

If `java -version` fails, install Java 11 or newer first.

If Java is installed but PyTerrier still does not detect it, set `JAVA_HOME` explicitly.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Start Jupyter Notebook from the project root:

```bash
jupyter notebook
```

Then use the notebooks in this order:

1. `notebooks/experiments.ipynb`
2. `notebooks/analysis.ipynb`

`experiments.ipynb` generates the cached result files in `results/cache/`.

`analysis.ipynb` reads those cached files and performs the evaluation and query-level analysis.

## How The Code Is Organized

The project is split into three parts:

- `src/config.py`: shared experiment configuration used by both notebooks
- `src/pipelines.py`: reusable helper code for building retrieval pipelines, caching ranking outputs, and comparing systems per query
- `src/analysis.py`: helper code for lightweight query-level analysis, gain labels, and result summaries
- `notebooks/experiments.ipynb`: notebook for loading data, building pipelines, and writing cached runs
- `notebooks/analysis.ipynb`: notebook for evaluation, statistical tests, and query-level analysis

`src/config.py` is used for:

- keeping `FAST_MODE`, `FB_TERMS`, `FB_VALUES`, `RERANK_K`, and related settings in one place
- making sure `experiments.ipynb` and `analysis.ipynb` use the same configuration

`src/pipelines.py` is used for:

- building `BM25`, `BM25 + RM3`, `BM25 + TAS-B`, and `BM25 + RM3 + TAS-B`
- caching pipeline outputs in `results/cache/` so expensive runs do not have to be repeated
- evaluating saved runs without recomputing them
- making per-query comparison tables

`src/analysis.py` is used for:

- adding simple query features
- labeling queries as helped, hurt, or neutral
- summarizing which systems perform best per query

`notebooks/experiments.ipynb` is used for:

- loading the datasets and Terrier index
- choosing a fast or full configuration
- calling the shared pipeline helpers
- writing cached run files for DL19 and DL20

`notebooks/analysis.ipynb` is used for:

- loading cached run files from `results/cache/`
- evaluating aggregate performance
- running statistical tests
- comparing different `fb_terms` settings
- inspecting top improved and harmed queries

## Recommended Workflow

The notebooks are designed so you do not need to rerun everything every time.

1. Set the experiment parameters once in `src/config.py`
2. Run `notebooks/experiments.ipynb` to generate the cached ranking outputs in `results/cache/`
3. Run `notebooks/analysis.ipynb` when you want to compare systems or inspect specific queries

Use `FAST_MODE = True` while developing. This runs fewer topics and uses a smaller reranking depth. Switch to `FAST_MODE = False` only for the final full experiment run.

The `results/cache/` directory is created locally and is not tracked in Git. Each user should run the experiments notebook once to generate their own cached result files.

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
- `irds:msmarco-passage/trec-dl-2019/judged` for DL19 qrels
- `irds:msmarco-passage/trec-dl-2019` for DL19 topics
- `irds:msmarco-passage/trec-dl-2020/judged` for DL20 topics and qrels

---

## Repository Structure

```text
notebooks/
    experiments.ipynb    # runs experiments and writes cached result files
    analysis.ipynb       # reads cached runs and performs evaluation/analysis

results/
    cache/               # cached ranking outputs for reuse across runs

src/
    config.py            # shared configuration for both notebooks
    pipelines.py         # reusable pipeline, caching, and per-query helpers
    analysis.py          # query-level analysis and summary helpers

requirements.txt         # dependencies
README.md
```
