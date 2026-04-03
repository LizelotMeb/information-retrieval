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

We use the **Vaswani dataset**, provided via PyTerrier (`pt.get_dataset("vaswani")`), which includes:
- documents (corpus)
- queries (topics)
- relevance judgments (qrels)

---

## Repository Structure

```text
notebooks/
    experiments.ipynb    # main notebook with all experiments

results/
    vaswani_index/       # generated index (not tracked in Git)

src/                     # optional helper code

requirements.txt         # dependencies
README.md
