"""Đối soát và đo Trie Maximum Matching với tra cứu bằng ``set``.

Câu hỏi cần trả lời: với **đúng cùng thuật toán, cùng từ điển và cùng đầu vào**,
cấu trúc Trie (dò cạnh theo âm tiết) có nhanh hơn tra cứu bằng hash set (băm
tuple âm tiết) hay không? Benchmark này **không giả định** Trie nhanh hơn; nó đo
thật rồi báo số.

Nguyên tắc:

* **Oracle / parity gate:** một ``SetMaximumMatching`` được viết độc lập, không
  gọi vào ``engine.tokenizer.trie``. Hai bên phải cho ra token giống hệt nhau cho
  cả FMM, BMM và BiMM trên mọi câu. Nếu lệch dù một token, benchmark **từ chối
  ghi kết quả** và trả mã lỗi: số đo chỉ có nghĩa khi hai bên cùng hành vi.
* **Cùng điều kiện:** mỗi vòng lặp đo xen kẽ Trie và Set trên cùng một câu, và
  đảo thứ tự theo vòng để giảm thiên lệch cache/thứ tự.
* **Đầu vào cố định:** mẫu phân đoạn 150 câu trong ``data/gold`` và một lát cắt
  corpus xác định theo manifest; cả hai đều kiểm SHA256 trước khi đo.
* **150 nhãn vàng vẫn pending:** benchmark chỉ đọc ``sentence_text`` làm đầu vào,
  tuyệt đối không đọc/ghi nhãn, không tự gán nhãn, không sửa ``data/gold``.
* Bộ nhớ đo bằng ``tracemalloc`` (heap Python) khi dựng từng cấu trúc, có ghi rõ
  đây là số đo heap Python chứ không phải RSS toàn tiến trình.

Dùng::

    python -m benchmarks.benchmark_lookup
    python -m benchmarks.benchmark_lookup --repeat 5 --warmup 1 --corpus-docs 100
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.tokenizer.max_matching import (  # noqa: E402
    DEFAULT_DICTIONARY,
    MaximumMatching,
    dictionary_metadata,
)
from engine.tokenizer.preprocessor import normalize_segments  # noqa: E402
from evaluator.metrics import percentile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "data/gold/segmentation_sample.jsonl"
DEFAULT_SAMPLE_MANIFEST = ROOT / "data/gold/segmentation_manifest.json"
DEFAULT_CORPUS = ROOT / "data/corpus.jsonl"
DEFAULT_CORPUS_MANIFEST = ROOT / "data/corpus_manifest.json"
DEFAULT_OUT = ROOT / "results/lookup"

METHODS = ("fmm", "bmm", "bimm")
JOINER = "_"

DEFAULT_REPEAT = 5
DEFAULT_WARMUP = 1
DEFAULT_CORPUS_DOCS = 100

#: Số ví dụ lệch tối đa ghi vào payload, để không phình báo cáo.
MAX_MISMATCH_EXAMPLES = 10


class LookupBenchmarkError(ValueError):
    """Đầu vào không cố định hoặc hai cấu trúc cho ra output khác nhau."""


# --------------------------------------------------------------------------- #
# Nguồn dữ liệu cố định
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> List[Mapping]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sample_texts(sample_path: Path = DEFAULT_SAMPLE,
                      manifest_path: Path = DEFAULT_SAMPLE_MANIFEST) -> Dict[str, object]:
    """Đọc đúng 150 câu mẫu; kiểm SHA256 theo manifest, không chạm vào nhãn."""
    if not sample_path.is_file() or not manifest_path.is_file():
        raise LookupBenchmarkError(
            f"Thiếu {sample_path} hoặc {manifest_path}; chạy "
            "python -m scripts.prepare_segmentation trước")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(sample_path)
    if digest != manifest.get("sample_sha256"):
        raise LookupBenchmarkError("Sample SHA256 không khớp manifest; đầu vào không cố định")
    # Only sentence_text is used. Labels are never read, written or inferred.
    texts = [str(record["sentence_text"]) for record in _jsonl(sample_path)]
    if len(texts) != manifest.get("n_samples"):
        raise LookupBenchmarkError("Số câu mẫu không khớp manifest")
    return {
        "name": "sample",
        "description": "150 câu phân đoạn (15/lĩnh vực) từ data/gold, nhãn vẫn pending",
        "source": "data/gold/segmentation_sample.jsonl",
        "sha256": digest,
        "labels_status": manifest.get("status"),
        "labels_used": False,
        "texts": texts,
    }


def load_corpus_texts(corpus_path: Path = DEFAULT_CORPUS,
                      manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
                      limit: int = DEFAULT_CORPUS_DOCS) -> Dict[str, object]:
    """Lấy ``limit`` tài liệu đầu theo thứ tự file (xác định) làm lát cắt corpus."""
    if not corpus_path.is_file():
        raise LookupBenchmarkError(f"Thiếu corpus {corpus_path}")
    if not manifest_path.is_file():
        raise LookupBenchmarkError(
            f"Thiếu manifest khóa {manifest_path}; từ chối đo vì không kiểm được corpus")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("corpus_sha256") or manifest.get("sha256")
    if not expected:
        raise LookupBenchmarkError(
            f"Manifest {manifest_path} thiếu corpus_sha256/sha256; từ chối đo")
    digest = sha256_file(corpus_path)
    if expected != digest:
        raise LookupBenchmarkError("Corpus SHA256 không khớp manifest khóa")
    records = _jsonl(corpus_path)
    if limit <= 0 or limit > len(records):
        raise LookupBenchmarkError(f"corpus-docs phải trong 1..{len(records)}")
    subset = records[:limit]
    # Deterministic slice: file order, content only, so the workload is reproducible.
    texts = [str(record.get("content", "")) for record in subset]
    return {
        "name": "corpus",
        "description": f"{limit} tài liệu đầu của corpus khóa (thứ tự file, xác định)",
        "source": "data/corpus.jsonl",
        "sha256": digest,
        "slice_rule": f"first {limit} records by file order",
        "labels_used": False,
        "texts": texts,
    }


# --------------------------------------------------------------------------- #
# Cấu trúc Set: viết độc lập, không dùng Trie
# --------------------------------------------------------------------------- #

class SetMaximumMatching:
    """Maximum Matching dùng ``set`` các tuple âm tiết để tra cứu.

    Độc lập hoàn toàn với ``engine.tokenizer.trie``: mọi lần tra cứu là một phép
    băm tuple. Thuật toán (FMM/BMM/BiMM, biên dấu câu, quy tắc hòa của BiMM) phải
    trùng khớp với ``MaximumMatching`` để làm oracle đối soát.
    """

    def __init__(self, entries: Iterable[str], max_syllables: int | None = None) -> None:
        self.entries = frozenset(tuple(str(word).split()) for word in entries
                                 if str(word).strip())
        self.words_count = len(self.entries)
        if max_syllables is None:
            self.max_syllables = max((len(entry) for entry in self.entries), default=1)
        else:
            self.max_syllables = int(max_syllables)
        if self.max_syllables < 1:
            raise ValueError("max_syllables must be >= 1")

    def _layout(self, text: str) -> tuple[list[str], list[int]]:
        syllables: list[str] = []
        segment_ends: list[int] = []
        for segment in normalize_segments(text):
            syllables.extend(segment.split(" "))
            segment_ends.append(len(syllables))
        return syllables, segment_ends

    def _hit(self, syllables: Sequence[str], start: int, end: int) -> bool:
        return tuple(syllables[start:end]) in self.entries

    def contains(self, token: str) -> bool:
        return tuple(token.split(JOINER)) in self.entries

    def decision_key(self, tokens: Sequence[str]) -> tuple[int, int, int]:
        singles = sum(1 for token in tokens if JOINER not in token)
        oov = sum(1 for token in tokens if not self.contains(token))
        return (len(tokens), singles, oov)

    def tokenize(self, text: str, method: str = "bimm") -> list[str]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not isinstance(text, str) or not text:
            return []
        layout = self._layout(text)
        if method == "fmm":
            return self._fmm(layout)
        if method == "bmm":
            return self._bmm(layout)
        return self._bimm(layout)

    def _window_limit(self, start: int, end: int, total: int) -> int:
        return max(min(self.max_syllables, end - start, total - start), 1)

    def _fmm(self, layout) -> list[str]:
        syllables, segment_ends = layout
        total = len(syllables)
        tokens: list[str] = []
        start = 0
        while start < total:
            segment_end = segment_ends[bisect.bisect_left(segment_ends, start + 1)]
            limit = self._window_limit(start, segment_end, total)
            matched_end = None
            for size in range(limit, 0, -1):
                if self._hit(syllables, start, start + size):
                    matched_end = start + size
                    break
            if matched_end is None:
                tokens.append(syllables[start])
                start += 1
            else:
                tokens.append(JOINER.join(syllables[start:matched_end]))
                start = matched_end
        return tokens

    def _bmm(self, layout) -> list[str]:
        syllables, segment_ends = layout
        total = len(syllables)
        backwards: list[str] = []
        end_cursor = total
        while end_cursor > 0:
            segment_index = bisect.bisect_left(segment_ends, end_cursor)
            segment_start = segment_ends[segment_index - 1] if segment_index else 0
            limit = self._window_limit(segment_start, end_cursor, total)
            matched_start = None
            for size in range(limit, 0, -1):
                start = end_cursor - size
                if start < segment_start:
                    break
                if self._hit(syllables, start, end_cursor):
                    matched_start = start
                    break
            if matched_start is None:
                backwards.append(syllables[end_cursor - 1])
                end_cursor -= 1
            else:
                backwards.append(JOINER.join(syllables[matched_start:end_cursor]))
                end_cursor = matched_start
        backwards.reverse()
        return backwards

    def _bimm(self, layout) -> list[str]:
        forward = self._fmm(layout)
        backward = self._bmm(layout)
        if self.decision_key(backward) <= self.decision_key(forward):
            return backward
        return forward

    def stats(self) -> dict:
        return {"structure": "set(tuple)", "entries": self.words_count,
                "max_syllables": self.max_syllables}


# --------------------------------------------------------------------------- #
# Dựng cấu trúc + đo bộ nhớ
# --------------------------------------------------------------------------- #

def load_dictionary_entries(path: Path = DEFAULT_DICTIONARY) -> List[str]:
    if not path.is_file():
        raise LookupBenchmarkError(f"Thiếu từ điển {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_matchers(entries: Sequence[str]) -> Dict[str, object]:
    """Dựng Trie và Set trên cùng danh sách mục; đo đỉnh heap Python khi dựng."""
    tracemalloc.start()
    trie_matcher = MaximumMatching(entries)
    _, trie_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    set_matcher = SetMaximumMatching(entries)
    _, set_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "trie": trie_matcher,
        "set": set_matcher,
        "memory": {
            "trie_build_peak_bytes": trie_peak,
            "set_build_peak_bytes": set_peak,
            "measurement": "tracemalloc peak during construction (Python heap only, not RSS)",
        },
        "stats": {"trie": trie_matcher.trie.stats(), "set": set_matcher.stats()},
    }


# --------------------------------------------------------------------------- #
# Oracle: đối soát output
# --------------------------------------------------------------------------- #

def parity_report(matchers: Mapping[str, object], texts: Sequence[str],
                  methods: Sequence[str] = METHODS) -> Dict[str, object]:
    """Đối chiếu từng token của Trie và Set; đây là oracle cứng của benchmark."""
    trie, set_matcher = matchers["trie"], matchers["set"]
    mismatches: List[Dict[str, object]] = []
    compared = 0
    n_mismatches = 0
    for method in methods:
        for index, text in enumerate(texts):
            expected = trie.tokenize(text, method)
            actual = set_matcher.tokenize(text, method)
            compared += 1
            if expected != actual:
                n_mismatches += 1
                if len(mismatches) < MAX_MISMATCH_EXAMPLES:
                    mismatches.append({
                        "method": method, "text_index": index, "text": text[:160],
                        "trie": expected, "set": actual,
                    })
    return {"ok": n_mismatches == 0, "comparisons": compared,
            "n_mismatches": n_mismatches, "examples": mismatches,
            "checked_methods": list(methods), "n_texts": len(texts)}


# --------------------------------------------------------------------------- #
# Đo thời gian
# --------------------------------------------------------------------------- #

def _summarize(nanoseconds: Sequence[int]) -> Dict[str, float]:
    milliseconds = sorted(value / 1e6 for value in nanoseconds)
    total = float(sum(milliseconds))
    return {
        "mean_ms": float(statistics.fmean(milliseconds)) if milliseconds else 0.0,
        "median_ms": float(statistics.median(milliseconds)) if milliseconds else 0.0,
        "p95_ms": percentile(milliseconds, 0.95),
        "p99_ms": percentile(milliseconds, 0.99),
        "min_ms": milliseconds[0] if milliseconds else 0.0,
        "max_ms": milliseconds[-1] if milliseconds else 0.0,
        "stdev_ms": float(statistics.pstdev(milliseconds)) if len(milliseconds) > 1 else 0.0,
        "total_ms": total,
        "n_measurements": len(milliseconds),
        "texts_per_second": (len(milliseconds) / (total / 1000.0)) if total else 0.0,
    }


def measure_pair(trie, set_matcher, texts: Sequence[str], method: str,
                 repeat: int, warmup: int) -> Dict[str, Dict[str, float]]:
    """Đo xen kẽ Trie/Set trên cùng câu; đảo thứ tự mỗi vòng để giảm thiên lệch."""
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    for _ in range(max(warmup, 0)):
        for text in texts:
            trie.tokenize(text, method)
            set_matcher.tokenize(text, method)
    trie_ns: List[int] = []
    set_ns: List[int] = []
    for round_index in range(repeat):
        trie_first = round_index % 2 == 0
        for text in texts:
            order = (("trie", trie, trie_ns), ("set", set_matcher, set_ns))
            if not trie_first:
                order = tuple(reversed(order))
            for _name, matcher, sink in order:
                started = time.perf_counter_ns()
                matcher.tokenize(text, method)
                sink.append(time.perf_counter_ns() - started)
    trie_summary = _summarize(trie_ns)
    set_summary = _summarize(set_ns)
    ratio = (set_summary["mean_ms"] / trie_summary["mean_ms"]
             if trie_summary["mean_ms"] else 0.0)
    return {
        "trie": trie_summary,
        "set": set_summary,
        "set_over_trie_mean_ratio": ratio,
        "trie_faster_on_mean": trie_summary["mean_ms"] < set_summary["mean_ms"],
    }


def measure_workload(workload: Mapping[str, object], matchers: Mapping[str, object],
                     methods: Sequence[str], repeat: int, warmup: int) -> Dict[str, object]:
    texts = workload["texts"]
    measurements = {
        method: measure_pair(matchers["trie"], matchers["set"], texts, method, repeat, warmup)
        for method in methods
    }
    return {
        "name": workload["name"],
        "description": workload["description"],
        "source": workload["source"],
        "sha256": workload["sha256"],
        "n_texts": len(texts),
        "total_chars": sum(len(text) for text in texts),
        "measurements": measurements,
    }


# --------------------------------------------------------------------------- #
# Chạy + ghi kết quả
# --------------------------------------------------------------------------- #

def environment() -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "timer": "time.perf_counter_ns",
    }


def run(*, sample_path: Path = DEFAULT_SAMPLE,
        sample_manifest: Path = DEFAULT_SAMPLE_MANIFEST,
        corpus_path: Path = DEFAULT_CORPUS,
        corpus_manifest: Path = DEFAULT_CORPUS_MANIFEST,
        dictionary_path: Path = DEFAULT_DICTIONARY,
        methods: Sequence[str] = METHODS,
        repeat: int = DEFAULT_REPEAT, warmup: int = DEFAULT_WARMUP,
        corpus_docs: int = DEFAULT_CORPUS_DOCS,
        out: Path = DEFAULT_OUT) -> Dict[str, object]:
    """Đo cả hai đầu vào. Ném ``LookupBenchmarkError`` nếu parity thất bại."""
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise LookupBenchmarkError(f"Method không hợp lệ: {unknown}")

    sample = load_sample_texts(sample_path, sample_manifest)
    corpus = load_corpus_texts(corpus_path, corpus_manifest, corpus_docs)
    entries = load_dictionary_entries(dictionary_path)
    built = build_matchers(entries)

    parity = parity_report(built, list(sample["texts"]) + list(corpus["texts"]), methods)
    if not parity["ok"]:
        first = parity["examples"][0]
        raise LookupBenchmarkError(
            "Parity thất bại giữa Trie và Set ("
            f"method={first['method']}, text_index={first['text_index']}): "
            f"trie={first['trie']} != set={first['set']}. Từ chối ghi kết quả.")

    workloads = {
        workload["name"]: measure_workload(workload, built, methods, repeat, warmup)
        for workload in (sample, corpus)
    }

    dictionary = dictionary_metadata()
    payload = {
        "schema_version": 1,
        "component": "benchmarks.benchmark_lookup",
        "question": "Trie Maximum Matching có nhanh hơn tra cứu set trên cùng đầu vào?",
        "measurement_status": "measured",
        "parity": parity,
        "environment": environment(),
        "parameters": {
            "repeat_per_text": repeat,
            "warmup_iterations": warmup,
            "methods": list(methods),
            "corpus_docs": corpus_docs,
            "measurement_order": "xen kẽ Trie/Set trên từng câu, đảo thứ tự mỗi vòng",
        },
        "dictionary": {
            "path": dictionary["dictionary_path"],
            "sha256": dictionary["dictionary_sha256"],
            "entries": dictionary["dictionary_entries"],
        },
        "memory": built["memory"],
        "structure_stats": built["stats"],
        "workloads": workloads,
        "labels": {
            "sample_labels_status": sample["labels_status"],
            "labels_read_or_written": False,
            "note": "Timing không cần nhãn. 150 nhãn vàng vẫn pending; benchmark không "
                    "tự gán nhãn và không sửa data/gold.",
        },
        "notes": [
            "Hai cấu trúc dùng đúng cùng thuật toán, cùng từ điển và cùng chuẩn hóa; "
            "oracle parity phải đạt trước khi ghi số.",
            "Số bộ nhớ là đỉnh heap Python khi dựng cấu trúc, không phải RSS tiến trình.",
            "Không kết luận bên nào nhanh hơn ngoài tỉ lệ đo được; chênh lệch nhỏ hơn "
            "độ nhiễu của máy thì không đáng coi là thắng.",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "trie_vs_set.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(destination / "trie_vs_set.csv", payload)
    return payload


def _write_csv(path: Path, payload: Mapping[str, object]) -> None:
    columns = ("workload", "method", "implementation", "mean_ms", "median_ms", "p95_ms",
               "p99_ms", "min_ms", "max_ms", "stdev_ms", "n_measurements",
               "texts_per_second", "set_over_trie_mean_ratio", "trie_faster_on_mean")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for name, workload in payload["workloads"].items():
            for method, measured in workload["measurements"].items():
                for implementation in ("trie", "set"):
                    writer.writerow({
                        "workload": name, "method": method,
                        "implementation": implementation,
                        "set_over_trie_mean_ratio": measured["set_over_trie_mean_ratio"],
                        "trie_faster_on_mean": measured["trie_faster_on_mean"],
                        **measured[implementation],
                    })


def main(argv: Iterable[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="Số vòng lặp mỗi câu (mặc định 5)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Số lượt chạy toàn bộ workload trước khi đo (mặc định 1)")
    parser.add_argument("--corpus-docs", type=int, default=DEFAULT_CORPUS_DOCS,
                        help="Số tài liệu đầu của corpus dùng làm đầu vào (mặc định 100)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        payload = run(sample_path=args.sample, sample_manifest=args.sample_manifest,
                      corpus_path=args.corpus, corpus_manifest=args.corpus_manifest,
                      dictionary_path=args.dictionary, methods=args.methods,
                      repeat=args.repeat, warmup=args.warmup,
                      corpus_docs=args.corpus_docs, out=args.out)
    except LookupBenchmarkError as error:
        print(f"Không đo được: {error}", file=sys.stderr)
        return 2

    parity = payload["parity"]
    print(f"parity ok={parity['ok']} ({parity['comparisons']} phép đối soát, "
          f"mismatch={parity['n_mismatches']})")
    for name, workload in payload["workloads"].items():
        print(f"[{name}] {workload['n_texts']} văn bản, {workload['total_chars']} ký tự")
        for method, measured in workload["measurements"].items():
            trie, setm = measured["trie"], measured["set"]
            print(f"  {method:5s} trie mean={trie['mean_ms']:8.4f} p95={trie['p95_ms']:8.4f} | "
                  f"set mean={setm['mean_ms']:8.4f} p95={setm['p95_ms']:8.4f} | "
                  f"set/trie={measured['set_over_trie_mean_ratio']:.3f}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_CORPUS_DOCS",
    "DEFAULT_OUT",
    "DEFAULT_REPEAT",
    "DEFAULT_SAMPLE",
    "DEFAULT_WARMUP",
    "METHODS",
    "LookupBenchmarkError",
    "SetMaximumMatching",
    "build_matchers",
    "environment",
    "load_corpus_texts",
    "load_dictionary_entries",
    "load_sample_texts",
    "main",
    "measure_pair",
    "measure_workload",
    "parity_report",
    "run",
    "sha256_file",
]
