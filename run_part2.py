from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.config import AppConfig
from rag.document_sources.local_python import LocalPythonFunctionSource
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

    embedder = build_embedder(config)
    vector_store = build_vector_store(config)

    source = LocalPythonFunctionSource(config.paths.custom_functions_dir)
    ingestor = DocumentIngestor(embedder, vector_store, batch_size=config.embedding.batch_size)
    ingested = ingestor.ingest(source.iter_documents())
    print(f"[part2] Ingested or updated {ingested} custom documents.")
    print(f"[part2] Collection count: {vector_store.count()}")

    retriever = build_retriever(config, embedder, vector_store)
    generator = build_pipeline_generator(config)
    print(
        "[part2] Generator: "
        f"model={config.generator.model_name} "
        f"base_url={config.generator.base_url}"
    )
    runner = EvaluationRunner(retriever=retriever, generator=generator, grounding=GroundingHeuristic())

    targeted_queries = load_queries(config.paths.part2_targeted_queries, limit=args.query_limit)
    targeted_results = runner.run(targeted_queries)
    write_result_tables(
        targeted_results,
        jsonl_path=config.paths.results_dir / "part2_targeted_results.jsonl",
        csv_path=config.paths.results_dir / "part2_targeted_results.csv",
    )

    cross_queries = load_queries(config.paths.part2_cross_queries, limit=args.query_limit)
    cross_results = runner.run(cross_queries)
    write_result_tables(
        cross_results,
        jsonl_path=config.paths.results_dir / "part2_cross_corpus_results.jsonl",
        csv_path=config.paths.results_dir / "part2_cross_corpus_results.csv",
    )
    print(f"[part2] Wrote {len(targeted_results) + len(cross_results)} results to {config.paths.results_dir}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--query-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
