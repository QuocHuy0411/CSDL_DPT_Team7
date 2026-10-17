"""HTTP Server serving both the REST API and the interactive HTML interface.

Runs without any third-party web frameworks, using Python's built-in http.server.
Endpoints:
    GET  /                          -> serves app/index.html
    GET  /api/status                -> returns system metadata (corpus size, configs)
    GET  /api/search?q=...&config=C4&top_k=10&fallback=1
    GET  /api/explain?q=...&doc_id=...&config=C4
    GET  /api/tokenize?text=...&method=bimm
    GET  /api/tier1?text=...
    GET  /api/corpus_doc?doc_id=...
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
from pathlib import Path
import socketserver
import sys
import time
import urllib.parse
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.preprocessing import ROOT as ENGINE_ROOT
from scripts.search import SearchService, load_engine

SERVICES = {}
CORPUS_CACHE = {}


def get_service(config: str, fallback: bool = True):
    key = (config, fallback)
    if key not in SERVICES:
        index_dir = ROOT / f"data/indexes/{config}"
        if not index_dir.is_dir():
            raise FileNotFoundError(f"Chỉ mục cho {config} chưa được build tại {index_dir}")
        SERVICES[key] = SearchService(index_dir, enable_fallback=fallback)
    return SERVICES[key]


def get_corpus():
    global CORPUS_CACHE
    if not CORPUS_CACHE:
        corpus_path = ROOT / "data/corpus.jsonl"
        if corpus_path.is_file():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        doc = json.loads(line)
                        CORPUS_CACHE[doc["doc_id"]] = doc
    return CORPUS_CACHE


class SearchAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "app"), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self.serve_index_html()
            return

        if path.startswith("/api/"):
            self.handle_api(path, params)
            return

        # Fallback to static serving in app directory
        super().do_GET()

    def serve_index_html(self):
        index_file = ROOT / "app" / "index.html"
        if not index_file.is_file():
            self.send_error(404, "index.html not found")
            return
        content = index_file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def handle_api(self, path: str, params: dict):
        try:
            if path == "/api/status":
                self.api_status()
            elif path == "/api/search":
                self.api_search(params)
            elif path == "/api/explain":
                self.api_explain(params)
            elif path == "/api/tokenize":
                self.api_tokenize(params)
            elif path == "/api/tier1":
                self.api_tier1(params)
            elif path == "/api/corpus_doc":
                self.api_corpus_doc(params)
            else:
                self.send_json({"error": f"Endpoint not found: {path}"}, status=404)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def api_status(self):
        indexes = ROOT / "data/indexes"
        configs = [c for c in ("C1", "C2", "C3", "C4") if (indexes / c / "build_metadata.json").is_file()]
        corpus = get_corpus()
        self.send_json({
            "status": "online",
            "project": "Hệ thống truy xuất văn bản tiếng Việt sử dụng TF-IDF",
            "team": "Team 7 - Hệ CSDL Đa phương tiện",
            "available_configs": configs,
            "total_documents": len(corpus),
            "dictionary_entries": 69858,
            "latency_benchmarks": {
                "C1": {"mean_ms": 0.994, "p95_ms": 1.992, "size_bytes": 9621765, "reduction": "0%"},
                "C2": {"mean_ms": 0.463, "p95_ms": 1.034, "size_bytes": 6987156, "reduction": "-27.38%"},
                "C3": {"mean_ms": 0.772, "p95_ms": 1.737, "size_bytes": 8748327, "reduction": "-9.08%"},
                "C4": {"mean_ms": 0.487, "p95_ms": 1.355, "size_bytes": 6542944, "reduction": "-32.00%"}
            }
        })

    def api_search(self, params: dict):
        query = params.get("q", [""])[0].strip()
        config = params.get("config", ["C4"])[0]
        top_k = int(params.get("top_k", [10])[0])
        fallback = params.get("fallback", ["1"])[0] in ("1", "true", "True")

        if not query:
            self.send_json({"results": [], "query": "", "used_configuration": config, "elapsed_ms": 0})
            return

        service = get_service(config, fallback=fallback)
        start = time.perf_counter()
        resp = service.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Attach snippet and doc metadata
        corpus = get_corpus()
        results = []
        for hit in resp.get("results", []):
            doc_id = hit["doc_id"]
            doc = corpus.get(doc_id, {})
            results.append({
                "doc_id": doc_id,
                "score": hit["score"],
                "title": hit.get("title") or doc.get("title", "Không có tiêu đề"),
                "category": hit.get("category") or doc.get("category", "Tin tức"),
                "snippet": hit.get("snippet") or (doc.get("content", "")[:250] + "...")
            })

        self.send_json({
            "query": query,
            "requested_configuration": resp["requested_configuration"],
            "used_configuration": resp["used_configuration"],
            "fallback_used": resp["fallback_used"],
            "elapsed_ms": round(elapsed_ms, 3),
            "results": results
        })

    def api_explain(self, params: dict):
        query = params.get("q", [""])[0].strip()
        doc_id = params.get("doc_id", [""])[0].strip()
        config = params.get("config", ["C4"])[0]
        used_config = params.get("used_config", [config])[0]

        if not query or not doc_id:
            self.send_json({"error": "Thiếu q hoặc doc_id"}, status=400)
            return

        service = get_service(config, fallback=True)
        exp = service.explain(query, doc_id, used_configuration=used_config)
        self.send_json(exp)

    def api_tokenize(self, params: dict):
        text = params.get("text", [""])[0].strip()
        method = params.get("method", ["bimm"])[0].lower()
        if method not in ("fmm", "bmm", "bimm"):
            method = "bimm"

        from engine.tokenizer import max_matching, preprocessor
        norm_text = preprocessor.normalize(text)
        matcher = max_matching.MaximumMatching()
        tokens = matcher.tokenize(text, method=method)
        trace_steps = max_matching.trace(text, method=method)

        self.send_json({
            "original": text,
            "normalized": norm_text,
            "method": method,
            "tokens": tokens,
            "trace": trace_steps
        })

    def api_tier1(self, params: dict):
        text = params.get("text", [""])[0].strip()
        from engine.tokenizer import max_matching, stopwords
        matcher = max_matching.MaximumMatching()
        tokens = matcher.tokenize(text)
        flt = stopwords.get_default_filter()
        filtered = flt.filter(tokens)

        # Generate guard report
        kept_set = set(filtered)
        report = []
        for idx, t in enumerate(tokens, 1):
            protected = bool(flt.is_protected(t))
            removed = t not in kept_set
            if protected and t in flt.stopwords:
                reason = "Có trong hư từ tầng 1 nhưng được guard giữ lại"
            elif protected:
                reason = "Guard bảo vệ (giữ dù có thể là từ chức năng)"
            elif removed:
                reason = "Bị lọc tầng 1 (hư từ ngữ pháp)"
            else:
                reason = "Thực từ — giữ nguyên"
            report.append({
                "pos": idx,
                "token": t,
                "action": "loại" if removed else "giữ",
                "reason": reason,
                "is_guarded": protected
            })

        self.send_json({
            "original_tokens": tokens,
            "filtered_tokens": filtered,
            "report": report,
            "removed_count": len(tokens) - len(filtered)
        })

    def api_corpus_doc(self, params: dict):
        doc_id = params.get("doc_id", [""])[0].strip()
        corpus = get_corpus()
        if doc_id in corpus:
            self.send_json(corpus[doc_id])
        else:
            self.send_json({"error": "Không tìm thấy tài liệu"}, status=404)


def run_server(port: int = 8000, open_browser: bool = True):
    handler = SearchAPIHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n========================================================")
        print(f" Web Server Đồ Án HCSDL Đa Phương Tiện (Đề tài số 8)")
        print(f" URL: {url}")
        print(f" Bấm Ctrl + C để dừng server.")
        print(f"========================================================\n")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nĐang tắt server...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Web Server cho Đồ án TF-IDF")
    parser.add_argument("--port", type=int, default=8000, help="Cổng mạng (mặc định 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt")
    args = parser.parse_args()
    run_server(port=args.port, open_browser=not args.no_browser)
