"""Integration boundaries: preprocessing identity, isolated indexes and real fallback."""

import json
from pathlib import Path

import pytest

from engine.preprocessing import Preprocessor, baseline_tokenize, fixture_tier1, read_stoplist


def test_development_adapter_is_explicit_and_preserves_negation():
    processor = Preprocessor(tier1_enabled=True)
    assert processor("KHÔNG và chưa chẳng vì của") == ["không", "chưa", "chẳng", "vì"]
    assert processor.metadata()["status"] == "development_only"
    assert baseline_tokenize("Cơ sở dữ liệu, TCP/IP 2026") == ["cơ", "sở", "dữ", "liệu", "tcp", "ip", "2026"]
    with pytest.raises(ValueError, match="real M1"):
        Preprocessor(m1_verified=True)


def test_preprocessing_fingerprint_rejects_changed_state():
    processor = Preprocessor(tier1_enabled=True)
    metadata = processor.metadata()
    assert Preprocessor.from_metadata(metadata)("và bóng đá") == ["bóng", "đá"]
    metadata["fingerprint"] = "changed"
    with pytest.raises(ValueError, match="rebuild"):
        Preprocessor.from_metadata(metadata)


def test_static_list_is_actual_pinned_1942_entries():
    path = Path(__file__).resolve().parents[1] / "resources/static_stopwords.txt"
    assert len(read_stoplist(path)) == 1942
    assert "à_này" in read_stoplist(path)


def test_c4_uses_separate_c3_index_and_explanation(tmp_path):
    from engine.indexer.search_engine import SearchEngine
    from scripts.search import SearchService

    documents = [{"doc_id": "D1", "title": "Một", "content": "chung bóng đá", "category": "fixture"},
                 {"doc_id": "D2", "title": "Hai", "content": "chung máy tính", "category": "fixture"}]
    processor = Preprocessor(tier1_enabled=True)
    common = {"corpus_sha256": "fixture", "preprocessing": processor.metadata(),
              "preprocessing_status": "development_only", "n_documents": 2}
    for configuration, stops in (("C3", set()), ("C4", {"chung"})):
        metadata = {**common, "configuration": configuration}
        engine = SearchEngine(tokenizer=processor, stopwords=stops, metadata=metadata).fit(documents)
        directory = tmp_path / configuration
        engine.save(directory)
        (directory / "build_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pure = SearchService(tmp_path / "C4", enable_fallback=False)
    assert pure.search("chung")["results"] == []
    service = SearchService(tmp_path / "C4")
    response = service.search("chung")
    assert response["fallback_used"] and response["used_configuration"] == "C3"
    assert response["results"]
    explanation = service.explain("chung", response["results"][0]["doc_id"], used_configuration="C3")
    assert sum(row["contribution"] for row in explanation["contributions"]) == pytest.approx(
        response["results"][0]["score"])
    assert service.search("hoàntoànOOV")["results"] == []


def test_index_build_end_to_end_and_manifest_tampering(tmp_path):
    from scripts.build_indexes import build_indexes, load_corpus
    from scripts.search import load_engine

    documents = [{"doc_id": f"D{i}", "title": f"Fixture {i}", "content": text, "category": "fixture"}
                 for i, text in enumerate(["và chung bóng đá", "và chung máy tính", "chung thị trường"])]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents), encoding="utf-8")
    summary = build_indexes(corpus, tmp_path / "indexes")
    assert set(summary) == {"C1", "C2", "C3", "C4"}
    hashes = {row["metadata"]["corpus_sha256"] for row in summary.values()}
    assert len(hashes) == 1
    for configuration in summary:
        engine = load_engine(tmp_path / "indexes" / configuration)
        assert engine.metadata["configuration"] == configuration
        assert engine.metadata["preprocessing_status"] == "development_only"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpus_sha256": "tampered"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_corpus(corpus, manifest)
