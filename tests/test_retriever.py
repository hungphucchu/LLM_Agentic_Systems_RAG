from collections.abc import Sequence

from rag.domain.models import CodeDocument
from rag.retrieval.service import CodeRetriever, retrieve_top_k
from rag.vector_store import InMemoryVectorStore


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.vectors[query]


def test_retriever_returns_highest_cosine_match_first() -> None:
    docs = [
        CodeDocument(id="a", text="alpha", metadata={"repo": "repo", "path": "a.py", "func_name": "alpha"}),
        CodeDocument(id="b", text="beta", metadata={"repo": "repo", "path": "b.py", "func_name": "beta"}),
    ]
    vectors = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "find alpha": [0.9, 0.1]}
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(vectors)
    store.upsert(docs, embedder.embed_texts([doc.text for doc in docs]))

    retriever = CodeRetriever(embedder=embedder, vector_store=store, top_k=2)
    chunks = retriever.retrieve("find alpha")

    assert [chunk.document.id for chunk in chunks] == ["a", "b"]
    assert chunks[0].score > chunks[1].score


def test_retrieve_top_k_function_embeds_query_and_searches_store() -> None:
    docs = [
        CodeDocument(id="a", text="alpha", metadata={"repo": "repo", "path": "a.py", "func_name": "alpha"}),
        CodeDocument(id="b", text="beta", metadata={"repo": "repo", "path": "b.py", "func_name": "beta"}),
    ]
    vectors = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "find beta": [0.1, 0.9]}
    store = InMemoryVectorStore()
    embedder = FakeEmbedder(vectors)
    store.upsert(docs, embedder.embed_texts([doc.text for doc in docs]))

    chunks = retrieve_top_k("find beta", embedder, store, top_k=1)

    assert len(chunks) == 1
    assert chunks[0].document.id == "b"
