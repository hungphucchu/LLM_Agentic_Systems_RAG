# Code RAG

This repository implements **Track B, Code RAG** for CS 6263 Assignment 4.

The project builds an end-to-end Retrieval Augmented Generation pipeline over Python source code:

1. Load a starter corpus from Hugging Face `code_search_net` using the Python config.
2. Embed code/docstring records with `sentence-transformers/all-MiniLM-L6-v2`.
3. Store vectors in a persistent Chroma collection.
4. Retrieve top-k code chunks for natural language questions.
5. Generate grounded answers with citations formatted as `repo/path::func_name`.
6. Add local custom Python functions and rerun targeted and cross-corpus queries.

For the full report, generated result-table discussion, and reflection, see [REPORT.md](REPORT.md).

## Track Declaration

- **Chosen track:** Track B, Code RAG
- **Starter dataset:** `code_search_net` with config `python`
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** Chroma persistent client
- **Default top k:** 4
- **Generator model:** UTSA OpenAI-compatible `Qwen/Qwen3-8B` endpoint

## Project Layout

- `config/`: YAML configuration for models, paths, retrieval, and generation
- `data/custom_functions/`: local Python functions added for Part 2
- `data/queries/`: reproducible query sets for Part 1 and Part 2
- `src/rag/`: reusable RAG package
- `tests/`: unit tests for parsing, retrieval, prompting, and evaluation logic
- `artifacts/results/`: generated JSONL and CSV result tables
- `artifacts/chroma_code/`: generated persistent Chroma database, reproducible by scripts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Optional `.env` for an OpenAI-compatible endpoint:

```bash
UTSA_BASE_URL=http://149.165.171.140:8888/v1
UTSA_API_KEY=your_key_here
UTSA_MODEL=Qwen/Qwen3-8B
```

To use the UTSA Qwen endpoint, provide the endpoint values in `.env` or environment variables.

## Run

Part 1 populates the starter corpus and evaluates 10 baseline queries:

```bash
python run_part1.py
```

For a quicker smoke run:

```bash
python run_part1.py --starter-limit 50 --query-limit 2
```

Part 2 adds local Python functions and evaluates targeted plus cross-corpus queries:

```bash
python run_part2.py
```

Run tests:

```bash
python -m pytest
```
