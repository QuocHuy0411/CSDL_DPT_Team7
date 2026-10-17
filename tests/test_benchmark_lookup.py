"""Kiểm thử oracle và cơ chế từ chối của benchmark Trie-vs-Set.

Test chỉ kiểm hành vi cấu trúc: parity phải đúng trên fixture nhỏ, gate phải chặn
khi hai bên lệch, dữ liệu cố định phải được kiểm hash, và benchmark không được
đọc/ghi nhãn vàng. Không có test nào khẳng định bên nào nhanh hơn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import benchmark_lookup as bl

FIXTURE_WORDS = ["bóng", "đá", "bóng đá", "và", "cơ sở", "cơ sở dữ liệu", "dữ liệu"]
FIXTURE_TEXTS = [
    "bóng đá và bóng đá",
    "cơ sở dữ liệu rất tốt",
    "xin chào, đây là câu lạ!",
    "từ ngoài từ điển zzz qux",
    "",
    "...!!!",
    "câu 1. tiếp theo",
]


def _dataset(tmp_path: Path, sentences=None, documents=None):
    sentences = sentences or ["bóng đá và bóng đá", "cơ sở dữ liệu"]
    documents = documents or ["bóng đá và bóng đá", "cơ sở dữ liệu rất tốt"]
    gold = tmp_path / "gold"
    gold.mkdir()
    sample_path = gold / "segmentation_sample.jsonl"
    sample_path.write_text("".join(
        json.dumps({"sample_id": f"SG{index:03d}", "sentence_text": text,
                    "sentence_sha256": "sha"}, ensure_ascii=False) + "\n"
        for index, text in enumerate(sentences, 1)), encoding="utf-8")
    manifest = {"sample_sha256": bl.sha256_file(sample_path), "n_samples": len(sentences),
                "status": "pending_human_annotation"}
    (gold / "segmentation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("".join(
        json.dumps({"doc_id": f"d{index}", "content": text}, ensure_ascii=False) + "\n"
        for index, text in enumerate(documents)), encoding="utf-8")
    corpus_manifest = tmp_path / "corpus_manifest.json"
    corpus_manifest.write_text(
        json.dumps({"corpus_sha256": bl.sha256_file(corpus_path)}) + "\n", encoding="utf-8")

    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("\n".join(FIXTURE_WORDS) + "\n", encoding="utf-8")
    return {"sample": sample_path, "sample_manifest": gold / "segmentation_manifest.json",
            "corpus": corpus_path, "corpus_manifest": corpus_manifest, "dictionary": dictionary}


# --------------------------------------------------------------------------- #
# Oracle: Set không dùng Trie nhưng phải cho cùng output
# --------------------------------------------------------------------------- #

def test_set_matcher_matches_trie_for_all_methods():
    trie = bl.MaximumMatching(FIXTURE_WORDS)
    set_matcher = bl.SetMaximumMatching(FIXTURE_WORDS)
    for method in bl.METHODS:
        for text in FIXTURE_TEXTS:
            assert trie.tokenize(text, method) == set_matcher.tokenize(text, method)


def test_set_matcher_does_not_use_trie(monkeypatch):
    import engine.tokenizer.trie as trie_module

    class Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("SetMaximumMatching must not build a Trie")

    monkeypatch.setattr(trie_module, "Trie", Boom)
    matcher = bl.SetMaximumMatching(FIXTURE_WORDS)
    assert matcher.tokenize("bóng đá và bóng", "bimm") == ["bóng_đá", "và", "bóng"]


def test_set_matcher_respects_punctuation_boundary():
    matcher = bl.SetMaximumMatching(["học sinh"])
    # Punctuation forces a boundary so the two syllables cannot be fused.
    assert matcher.tokenize("học, sinh", "fmm") == ["học", "sinh"]
    assert matcher.tokenize("học sinh", "fmm") == ["học_sinh"]


def test_parity_report_flags_a_broken_set():
    built = bl.build_matchers(FIXTURE_WORDS)

    class Broken:
        def tokenize(self, text, method="bimm"):
            return ["WRONG"]

    built["set"] = Broken()
    report = bl.parity_report(built, ["bóng đá"], ["bimm"])
    assert report["ok"] is False
    assert report["n_mismatches"] == 1
    assert report["examples"][0]["set"] == ["WRONG"]


# --------------------------------------------------------------------------- #
# Đầu vào cố định
# --------------------------------------------------------------------------- #

def test_load_sample_texts_rejects_hash_mismatch(tmp_path):
    paths = _dataset(tmp_path)
    manifest = json.loads(paths["sample_manifest"].read_text(encoding="utf-8"))
    manifest["sample_sha256"] = "0" * 64
    paths["sample_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(bl.LookupBenchmarkError, match="SHA256"):
        bl.load_sample_texts(paths["sample"], paths["sample_manifest"])


def test_load_sample_reports_labels_unused(tmp_path):
    paths = _dataset(tmp_path)
    loaded = bl.load_sample_texts(paths["sample"], paths["sample_manifest"])
    assert loaded["texts"] == ["bóng đá và bóng đá", "cơ sở dữ liệu"]
    assert loaded["labels_used"] is False
    assert loaded["labels_status"] == "pending_human_annotation"


def test_load_corpus_slice_is_deterministic(tmp_path):
    paths = _dataset(tmp_path, documents=["a b c", "d e f", "g h i"])
    first = bl.load_corpus_texts(paths["corpus"], paths["corpus_manifest"], limit=2)
    second = bl.load_corpus_texts(paths["corpus"], paths["corpus_manifest"], limit=2)
    assert first["texts"] == ["a b c", "d e f"] == second["texts"]
    with pytest.raises(bl.LookupBenchmarkError, match="corpus-docs"):
        bl.load_corpus_texts(paths["corpus"], paths["corpus_manifest"], limit=9)


def test_load_corpus_rejects_unlocked_corpus(tmp_path):
    paths = _dataset(tmp_path)
    paths["corpus"].write_text('{"doc_id": "x", "content": "khac"}\n', encoding="utf-8")
    with pytest.raises(bl.LookupBenchmarkError, match="SHA256"):
        bl.load_corpus_texts(paths["corpus"], paths["corpus_manifest"], limit=1)


def test_load_corpus_rejects_missing_manifest(tmp_path):
    paths = _dataset(tmp_path)
    missing = tmp_path / "nope_manifest.json"
    with pytest.raises(bl.LookupBenchmarkError, match="manifest"):
        bl.load_corpus_texts(paths["corpus"], missing, limit=1)


def test_load_corpus_rejects_manifest_without_hash(tmp_path):
    paths = _dataset(tmp_path)
    paths["corpus_manifest"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(bl.LookupBenchmarkError, match="corpus_sha256"):
        bl.load_corpus_texts(paths["corpus"], paths["corpus_manifest"], limit=1)


# --------------------------------------------------------------------------- #
# Chạy end-to-end
# --------------------------------------------------------------------------- #

def test_run_writes_results_with_parity_true(tmp_path):
    paths = _dataset(tmp_path)
    out = tmp_path / "results"
    payload = bl.run(sample_path=paths["sample"], sample_manifest=paths["sample_manifest"],
                     corpus_path=paths["corpus"], corpus_manifest=paths["corpus_manifest"],
                     dictionary_path=paths["dictionary"], methods=["fmm", "bimm"],
                     repeat=1, warmup=0, corpus_docs=2, out=out)
    assert payload["parity"]["ok"] is True
    assert payload["parity"]["comparisons"] == (2 + 2) * 2
    assert payload["labels"]["labels_read_or_written"] is False
    for method in ("fmm", "bimm"):
        measured = payload["workloads"]["sample"]["measurements"][method]
        assert measured["trie"]["n_measurements"] == 2
        assert measured["set"]["n_measurements"] == 2
    assert (out / "trie_vs_set.json").is_file()
    assert (out / "trie_vs_set.csv").is_file()


def test_run_refuses_and_writes_nothing_on_parity_failure(tmp_path, monkeypatch):
    paths = _dataset(tmp_path)
    real_build = bl.build_matchers

    def broken_build(entries):
        built = real_build(entries)

        class Broken:
            def tokenize(self, text, method="bimm"):
                return ["WRONG"]

        built["set"] = Broken()
        return built

    monkeypatch.setattr(bl, "build_matchers", broken_build)
    out = tmp_path / "results"
    with pytest.raises(bl.LookupBenchmarkError, match="Parity"):
        bl.run(sample_path=paths["sample"], sample_manifest=paths["sample_manifest"],
               corpus_path=paths["corpus"], corpus_manifest=paths["corpus_manifest"],
               dictionary_path=paths["dictionary"], methods=["bimm"],
               repeat=1, warmup=0, corpus_docs=2, out=out)
    assert not (out / "trie_vs_set.json").exists()


def test_run_rejects_unknown_method(tmp_path):
    paths = _dataset(tmp_path)
    with pytest.raises(bl.LookupBenchmarkError, match="Method"):
        bl.run(dictionary_path=paths["dictionary"], methods=["nope"], repeat=1, warmup=0,
               out=tmp_path / "results")


def test_measure_pair_interleaves_both_implementations():
    trie = bl.MaximumMatching(FIXTURE_WORDS)
    set_matcher = bl.SetMaximumMatching(FIXTURE_WORDS)
    measured = bl.measure_pair(trie, set_matcher, FIXTURE_TEXTS, "bimm", repeat=2, warmup=1)
    assert measured["trie"]["n_measurements"] == len(FIXTURE_TEXTS) * 2
    assert measured["set"]["n_measurements"] == len(FIXTURE_TEXTS) * 2
    assert measured["set_over_trie_mean_ratio"] >= 0.0


def test_benchmark_does_not_mutate_gold_labels():
    labels = bl.ROOT / "data/gold/segmentation_labels.jsonl"
    sample = bl.ROOT / "data/gold/segmentation_sample.jsonl"
    before = (bl.sha256_file(labels), bl.sha256_file(sample))
    loaded = bl.load_sample_texts()
    after = (bl.sha256_file(labels), bl.sha256_file(sample))
    assert before == after
    assert len(loaded["texts"]) == 150
