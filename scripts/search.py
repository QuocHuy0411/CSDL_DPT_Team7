"""Load a built configuration with matching preprocessing and optional C4 fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.preprocessing import Preprocessor, ROOT


def load_engine(directory: str | Path):
    from engine.indexer.search_engine import SearchEngine
    directory = Path(directory)
    metadata = json.loads((directory / "build_metadata.json").read_text(encoding="utf-8"))
    processor = Preprocessor.from_metadata(metadata["preprocessing"])
    engine = SearchEngine.load(directory, tokenizer=processor)
    if engine.metadata.get("corpus_sha256") != metadata["corpus_sha256"]:
        raise ValueError("Index and build metadata corpus hashes disagree")
    return engine


class SearchService:
    def __init__(self, directory: str | Path, *, enable_fallback: bool = True):
        self.directory = Path(directory)
        self.engine = load_engine(self.directory)
        self.fallback = None
        self.enable_fallback = enable_fallback

    def search(self, query: str, top_k: int = 10):
        results = self.engine.search(query, top_k=top_k)
        used = self.engine.metadata["configuration"]
        if (not results and self.enable_fallback and used == "C4" and query.strip()):
            if self.fallback is None:
                self.fallback = load_engine(self.directory.parent / "C3")
            if self.fallback.metadata["corpus_sha256"] != self.engine.metadata["corpus_sha256"]:
                raise ValueError("C3 fallback corpus mismatch")
            if self.fallback.metadata["preprocessing"] != self.engine.metadata["preprocessing"]:
                raise ValueError("C3 fallback preprocessing mismatch")
            results = self.fallback.search(query, top_k=top_k)
            if results:
                used = "C3"
        return {"requested_configuration": self.engine.metadata["configuration"],
                "used_configuration": used, "fallback_used": used != self.engine.metadata["configuration"],
                "preprocessing_status": self.engine.metadata["preprocessing_status"],
                "results": results}

    def explain(self, query: str, doc_id: str, *, used_configuration: str | None = None):
        if used_configuration == "C3" and self.engine.metadata["configuration"] == "C4":
            if self.fallback is None:
                self.fallback = load_engine(self.directory.parent / "C3")
            return self.fallback.explain(query, doc_id)
        return self.engine.explain(query, doc_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--configuration", choices=["C1", "C2", "C3", "C4"], default="C4")
    parser.add_argument("--indexes", type=Path, default=ROOT / "data/indexes")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    service = SearchService(args.indexes / args.configuration, enable_fallback=not args.no_fallback)
    response = service.search(args.query, args.top_k)
    if args.explain:
        response["explanations"] = [service.explain(args.query, hit["doc_id"],
                                    used_configuration=response["used_configuration"])
                                    for hit in response["results"]]
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
