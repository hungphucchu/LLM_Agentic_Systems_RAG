from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.config import AppConfig
from rag.document_sources.code_search_net import CodeSearchNetSource
from rag.evaluation_core.grounding import GroundingHeuristic
from rag.evaluation_core.result_tables import load_queries, write_result_tables
from rag.evaluation_core.runner import EvaluationRunner
from rag.factory import build_embedder, build_pipeline_generator, build_retriever, build_vector_store
from rag.ingestion import DocumentIngestor
from rag.project_paths import ensure_repo_on_path

ensure_repo_on_path()


def main() -> None:
    args = _parse_args()
    config = AppConfig.from_yaml(args.config)
    starter_limit = args.starter_limit or config.dataset.starter_size

    embedder = build_embedder(config)
    vector_store = build_vector_store(config)

    if not args.skip_ingest:
        source = CodeSearchNetSource(
            dataset_name=config.dataset.name,
            config_name=config.dataset.config,
            split=config.dataset.split,
        )
        ingestor = DocumentIngestor(embedder, vector_store, batch_size=config.embedding.batch_size)
        ingested = ingestor.ingest(source.iter_documents(limit=starter_limit), limit=starter_limit)
        print(f"[part1] Ingested or updated {ingested} starter documents.")

    print(f"[part1] Collection count: {vector_store.count()}")
    retriever = build_retriever(config, embedder, vector_store)
    generator = build_pipeline_generator(config)
    print(
        "[part1] Generator: "
        f"model={config.generator.model_name} "
        f"base_url={config.generator.base_url}"
    )
    runner = EvaluationRunner(retriever=retriever, generator=generator, grounding=GroundingHeuristic())
    queries = load_queries(config.paths.part1_queries, limit=args.query_limit)
    results = runner.run(queries)
    write_result_tables(
        results,
        jsonl_path=config.paths.results_dir / "part1_results.jsonl",
        csv_path=config.paths.results_dir / "part1_results.csv",
    )
    print(f"[part1] Wrote {len(results)} results to {config.paths.results_dir}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--starter-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
