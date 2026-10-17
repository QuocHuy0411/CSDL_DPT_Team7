"""Build isolated C1-C4 artifacts from one locked corpus. Run as a module."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from engine.preprocessing import (BASELINE, M1_TOKENIZER, M1_TIER1, Preprocessor,
                                  ROOT, protected_terms, read_stoplist)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_corpus(path: str | Path, manifest: str | Path | None = None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if len({d["doc_id"] for d in documents}) != len(documents):
        raise ValueError("Duplicate doc_id in corpus")
    if manifest is not None:
        metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
        expected = metadata.get("corpus_sha256", metadata.get("sha256"))
        if expected != digest:
            raise ValueError("Corpus hash does not match the locked manifest")
        counts = Counter(d["category"] for d in documents)
        if len(documents) != 2500 or len(counts) != 10 or set(counts.values()) != {250}:
            raise ValueError("Main corpus must contain exactly 250 documents in each of 10 categories")
    return documents, digest


def build_indexes(corpus: str | Path, output: str | Path, *,
                  manifest: str | Path | None = None,
                  tokenizer: str = BASELINE,
                  tier1_filter: str = "engine.preprocessing:fixture_tier1",
                  m1_verified: bool = False, resources: tuple[str, ...] = (),
                  static_stoplist: str | Path = ROOT / "resources/static_stopwords.txt",
                  configurations=("C1", "C2", "C3", "C4")):
    from engine.indexer.search_engine import SearchEngine
    from engine.indexer.tdv_engine import TDVEngine

    documents, corpus_hash = load_corpus(corpus, manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requested = set(configurations)
    if not requested or not requested <= {"C1", "C2", "C3", "C4"}:
        raise ValueError("Configurations must be C1, C2, C3 or C4")
    static = read_stoplist(static_stoplist)
    plain = Preprocessor(tokenizer, tier1_filter, False, m1_verified, resources)
    tier1 = Preprocessor(tokenizer, tier1_filter, True, m1_verified, resources)
    print(f"Tokenizing {len(documents)} documents ({plain.metadata()['status']})", flush=True)
    raw_tokens = [plain(d["content"]) for d in documents]
    tier1_tokens = [tier1(d["content"]) for d in documents]
    removed_by_tier1 = Counter()
    for before, after in zip(raw_tokens, tier1_tokens):
        removed_by_tier1.update(Counter(before) - Counter(after))
    tdv = None
    tdv_seconds = 0.0
    if "C4" in requested:
        print("Computing exact TDV on the full selected corpus", flush=True)
        started = time.perf_counter()
        tdv = TDVEngine().fit(tier1_tokens)
        tdv_seconds = time.perf_counter() - started
    guards = protected_terms(m1_verified)
    if m1_verified:
        # Mirror M1's conservative component guard for dimensions, not just query filtering.
        components = {part for guard in guards for part in guard.split("_") if part in guards}
        guards = guards | frozenset(t for tokens in tier1_tokens for t in tokens
                                   if any(p in components for p in t.split("_")))
    summary = {}
    for name in ("C1", "C2", "C3", "C4"):
        if name not in requested:
            continue
        started = time.perf_counter()
        processor = tier1 if name in {"C3", "C4"} else plain
        tokens = tier1_tokens if name in {"C3", "C4"} else raw_tokens
        stops = static if name == "C2" else (
            set(tdv.stopwords_) - guards if name == "C4" else set())
        preprocessing = processor.metadata()
        metadata = {
            "schema_version": 1, "configuration": name,
            "corpus_sha256": corpus_hash, "n_documents": len(documents),
            "category_counts": dict(Counter(d["category"] for d in documents)),
            "preprocessing": preprocessing,
            "preprocessing_status": preprocessing["status"],
            "official_preprocessing": m1_verified,
            "status": "official_preprocessing" if m1_verified else "provisional",
            "tokenizer": tokenizer, "tier1_filter": tier1_filter,
            "static_stoplist_source": "stopwords/vietnamese-stopwords@a453d389e1b52e20748ca83ddd8b0faebb04f5aa" if name == "C2" else None,
            "stoplist_count": len(stops),
            "stoplist_count_scope": "additional list after tokenizer/tier1 adapter",
            "tier1_observed_removed_term_count": len(removed_by_tier1) if name in {"C3", "C4"} else 0,
            "tier1_observed_removed_token_count": sum(removed_by_tier1.values()) if name in {"C3", "C4"} else 0,
            "stoplist_sha256": hashlib.sha256("\n".join(sorted(stops)).encode("utf-8")).hexdigest(),
            "stoplist_matched_vocabulary_count": len(set(t for ts in tokens for t in ts) & stops),
            "tf": "1+ln(f)", "idf": "ln((N+1)/(df+1))+1", "normalization": "L2",
            "tdv_seconds": tdv_seconds if name == "C4" else None,
            "tdv_guarded_terms": sorted(set(tdv.stopwords_) & guards) if name == "C4" else [],
            "fallback_policy": "optional_separate_C3_index_if_C4_query_vector_zero" if name == "C4" else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
        }
        engine = SearchEngine(tokenizer=processor, stopwords=stops, metadata=metadata)
        engine.fit(documents, tokenized_docs=tokens)
        destination = output / name
        engine.save(destination)
        # A stable shared metadata entrypoint, independent of engine persistence details.
        write_json(destination / "build_metadata.json", metadata)
        if name == "C4":
            write_json(destination / "tdv_scores.json", tdv.scores_)
            write_json(destination / "tdv_stopwords.json", sorted(stops))
            write_json(destination / "tdv_metadata.json", tdv.metadata_)
            write_json(destination / "tdv_df.json", dict(Counter(
                term for document in tier1_tokens for term in set(document))))
        summary[name] = {"build_seconds": time.perf_counter() - started,
                         "artifact_bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
                         "metadata": metadata}
        print(f"{name}: saved {destination}, {summary[name]['artifact_bytes']} bytes", flush=True)
    write_json(output / "build_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/corpus_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/indexes")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tier1-filter", default=None)
    parser.add_argument("--development-baseline", action="store_true",
                        help="Explicitly select the provisional syllable/fixture pipeline")
    parser.add_argument("--m1-verified", action="store_true",
                        help="Use only after validating the actual M1 Maximum Matching and tier1 implementation")
    parser.add_argument("--resource", action="append", default=[], help="M1 dictionary/stoplist resource to fingerprint")
    parser.add_argument("--static-stoplist", type=Path, default=ROOT / "resources/static_stopwords.txt")
    parser.add_argument("--configurations", nargs="+", choices=["C1", "C2", "C3", "C4"], default=["C1", "C2", "C3", "C4"])
    args = parser.parse_args()
    if args.development_baseline and args.m1_verified:
        parser.error("--development-baseline cannot be --m1-verified")
    tokenizer = args.tokenizer or (BASELINE if args.development_baseline else M1_TOKENIZER)
    tier1_filter = args.tier1_filter or (
        "engine.preprocessing:fixture_tier1" if args.development_baseline else M1_TIER1)
    verified = args.m1_verified or (
        not args.development_baseline and tokenizer == M1_TOKENIZER and tier1_filter == M1_TIER1)
    build_indexes(args.corpus, args.output, manifest=args.manifest,
                  tokenizer=tokenizer, tier1_filter=tier1_filter,
                  m1_verified=verified, resources=tuple(args.resource),
                  static_stoplist=args.static_stoplist, configurations=args.configurations)


if __name__ == "__main__":
    main()
