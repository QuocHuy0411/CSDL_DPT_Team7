"""Clean, validate and pin the Vietnamese dictionary used by the M1 tokenizer.

Input   : resources/dictionary_source/Viet74K.txt   (pinned upstream snapshot)
Output  : resources/dict.txt                        (one entry per line)
          resources/dict_stats.json                 (counts + provenance + hashes)
          resources/dictionary_source.json          (machine readable provenance)

The build is fully offline: it reads the vendored snapshot. `--fetch` re-downloads
the exact pinned commit from GitHub and refuses to continue if the SHA256 differs
from the recorded one, so a drifting upstream cannot silently change the dictionary.

Cleaning rules, applied in this order:
  1. drop the BOM, decode as UTF-8
  2. NFC normalization
  3. lowercase (str.lower, not str.casefold: casefold applies mappings such as
     ss for sharp s that Vietnamese does not have, which would merge distinct
     entries)
  4. REJECT the whole entry if any character is outside the Vietnamese letter set
     or the entry is otherwise malformed. Entries are rejected, never repaired:
     an entry is not partially stripped of characters, because that would invent
     a word the source does not contain.
  5. collapse whitespace runs to one space, strip
  6. drop empty entries, drop duplicates, sort for a stable file

Every rejected entry increments a named counter, so the stats file shows exactly
why the output is smaller than the input. Nothing is padded or invented to reach
a target count.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "resources/dictionary_source"
RAW_NAME = "Viet74K.txt"
DEFAULT_OUTPUT = ROOT / "resources/dict.txt"
DEFAULT_STATS = ROOT / "resources/dict_stats.json"
DEFAULT_PROVENANCE = ROOT / "resources/dictionary_source.json"
LICENSE_NAME = "LICENSE"

UPSTREAM_REPO = "https://github.com/duyet/vietnamese-wordlist"
UPSTREAM_COMMIT = "688b0953c8cdb21ee090ce80b83cefdb052fe2cd"
UPSTREAM_BRANCH = "master"
UPSTREAM_RAW_URL = f"https://raw.githubusercontent.com/duyet/vietnamese-wordlist/{UPSTREAM_COMMIT}/{RAW_NAME}"
UPSTREAM_LICENSE = "GPL-2.0"
UPSTREAM_AUTHOR = "Ho Ngoc Duc (Universitaet Leipzig)"
UPSTREAM_MIRROR_MAINTAINER = "Van-Duyet Le (duyet)"
EXPECTED_RAW_SHA256 = "b83d49308336f12ca46d06287d8db6e2cdc7a725d9a9dac56930051775fafb90"
EXPECTED_LICENSE_SHA256 = "c03cea027b4b40e4402fabd08557736727ec3d5bc54ad64ab6472de432198cad"

# Vietnamese letters, exactly. Anything else is not part of a dictionary entry.
VIETNAMESE_LETTERS = (
    "aăâbcdđeêghiklmnoôơpqrstuưvxy"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụưứừửữựýỳỷỹỵ"
)
VALID_ENTRY = re.compile(rf"^[{VIETNAMESE_LETTERS}]+(?: [{VIETNAMESE_LETTERS}]+)*$")
WHITESPACE_RUN = re.compile(r"\s+")
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    """Repo relative when possible, absolute otherwise.

    Test fixtures live in a temporary directory outside the repository, and
    Path.relative_to raises instead of degrading, so it cannot be used directly.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class SourceVerificationError(RuntimeError):
    """Raised when the vendored snapshot does not match the pinned commit."""


def verify_snapshot(raw_path: Path, license_path: Path, pinned: bool) -> dict:
    """Check the vendored snapshot and its licence before anything is written.

    This runs on every build, not only with --fetch, so a hand edited or truncated
    resources/dictionary_source/Viet74K.txt can never silently become the
    dictionary used by the tokenizer.
    """
    if not raw_path.is_file():
        raise SourceVerificationError(f"missing dictionary snapshot: {display_path(raw_path)}")
    digest = sha256_file(raw_path)
    result = {
        "path": display_path(raw_path),
        "sha256": digest,
        "bytes": raw_path.stat().st_size,
        "pinned": pinned,
    }
    if not pinned:
        result["note"] = ("custom source passed with --raw; this is NOT the pinned "
                          "Viet74K snapshot and carries no upstream provenance")
        return result
    if digest != EXPECTED_RAW_SHA256:
        raise SourceVerificationError(
            f"{display_path(raw_path)} sha256 {digest} does not match the pinned "
            f"{EXPECTED_RAW_SHA256}; refusing to build a dictionary from it")
    if not license_path.is_file():
        raise SourceVerificationError(f"missing licence file: {display_path(license_path)}")
    licence_digest = sha256_file(license_path)
    if licence_digest != EXPECTED_LICENSE_SHA256:
        raise SourceVerificationError(
            f"{display_path(license_path)} sha256 {licence_digest} does not match the "
            f"pinned {EXPECTED_LICENSE_SHA256}")
    result["license_path"] = display_path(license_path)
    result["license_sha256"] = licence_digest
    result["verified"] = True
    return result


def fetch_pinned(url: str, expected_sha256: str, destination: Path) -> str:
    """Download the pinned file and refuse if the bytes are not what we recorded."""
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise SourceVerificationError(
            f"refusing to use {url}: sha256 {digest} != pinned {expected_sha256}")
    destination.write_bytes(payload)
    return digest


def clean_entry(raw_line: str, counters: Counter) -> str | None:
    """Return the accepted dictionary entry, or None with a reason in `counters`.

    A malformed entry is rejected whole. Characters are never removed from the
    middle of an entry, because doing so would create a word the source never
    contained.
    """
    line = raw_line.replace("﻿", "").strip()
    if not line:
        counters["rejected_blank"] += 1
        return None
    entry = unicodedata.normalize("NFC", line).lower()
    entry = WHITESPACE_RUN.sub(" ", entry).strip()
    if not entry:
        counters["rejected_empty_after_whitespace"] += 1
        return None
    if not VALID_ENTRY.match(entry):
        counters["rejected_invalid_entries"] += 1
        return None
    return entry


def clean_dictionary(raw_path: Path, output_path: Path, stats_path: Path,
                    provenance_path: Path) -> dict:
    raw_text = raw_path.read_text(encoding="utf-8")
    lines = raw_text.split("\n")
    counters: Counter = Counter()
    seen: set[str] = set()
    entries: list[str] = []
    duplicates = 0
    for line in lines:
        entry = clean_entry(line, counters)
        if entry is None:
            continue
        if entry in seen:
            duplicates += 1
            continue
        seen.add(entry)
        entries.append(entry)
    entries.sort()

    syllable_counts = Counter(len(entry.split(" ")) for entry in entries)
    total = len(entries)
    within_four = sum(count for size, count in syllable_counts.items() if size <= 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(entry + "\n")

    provenance = (json.loads(provenance_path.read_text(encoding="utf-8"))
                  if provenance_path.is_file() else None)
    stats = {
        "schema_version": SCHEMA_VERSION,
        "total_words": total,
        "syllable_distribution": {str(k): v for k, v in sorted(syllable_counts.items())},
        "max_syllables": max(syllable_counts) if syllable_counts else 0,
        "words_with_at_most_4_syllables": within_four,
        "share_within_4_syllables_percent": round(100.0 * within_four / total, 4) if total else 0.0,
        "single_syllable_words": syllable_counts.get(1, 0),
        "multi_syllable_words": total - syllable_counts.get(1, 0),
        "input": provenance.get("vendored_snapshot") if provenance else None,
        "output": {
            "path": display_path(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "rejected_or_removed": {
            "blank_lines": counters["rejected_blank"],
            "empty_after_whitespace": counters["rejected_empty_after_whitespace"],
            "invalid_entries_rejected_whole": counters["rejected_invalid_entries"],
            "duplicate_entries": duplicates,
            "total_not_kept": len(lines) - total,
        },
        "cleaning_rules": [
            "strip BOM, decode UTF-8",
            "unicode NFC",
            "lowercase (str.lower, not str.casefold)",
            "REJECT the entry whole if it contains any character outside the Vietnamese "
            "letter set; characters are never deleted from inside an entry",
            "collapse whitespace runs to a single space and strip",
            "drop empty entries, drop duplicates, sort ascending",
        ],
        "window_note": (
            "max_syllables in this file is the longest entry in the dictionary and is the "
            "default sliding window bound used by engine.tokenizer.max_matching"
        ),
        "source": provenance,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return stats


def write_provenance(path: Path, raw_path: Path, license_path: Path, pinned: bool) -> dict:
    if not pinned:
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "name": None,
            "pinned": False,
            "upstream_repository": None,
            "upstream_commit": None,
            "license": None,
            "vendored_snapshot": {
                "path": display_path(raw_path),
                "sha256": sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            },
            "note": "custom source passed with --raw; it is NOT the pinned Viet74K "
                    "snapshot, so no upstream repository, commit or licence is claimed.",
        }
        path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return provenance
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "name": "Viet74K",
        "pinned": True,
        "upstream_repository": UPSTREAM_REPO,
        "upstream_branch": UPSTREAM_BRANCH,
        "pinned_commit": UPSTREAM_COMMIT,
        "raw_url": UPSTREAM_RAW_URL,
        "license": UPSTREAM_LICENSE,
        "license_evidence": {
            "github_api_license_field": "gpl-2.0",
            "license_file_vendored_at":
                display_path(license_path) if license_path.is_file() else None,
            "license_sha256": sha256_file(license_path) if license_path.is_file() else None,
            "license_first_line": ("GNU GENERAL PUBLIC LICENSE"
                                   if license_path.is_file() else None),
            "note": "The LICENSE file is vendored next to the data snapshot, so the grant "
                    "travels with the data.",
        },
        "original_author": UPSTREAM_AUTHOR,
        "mirror_maintainer": UPSTREAM_MIRROR_MAINTAINER,
        "vendored_snapshot": {
            "path": display_path(raw_path),
            "sha256": sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
        },
        "why_this_source": [
            "The dictionary named in the project plan (roy-a/Roy_VnTokenizer) carries no "
            "LICENSE file and the GitHub API reports license=null, so its word list cannot "
            "be redistributed.",
            "Viet74K is a real, widely reused Vietnamese word list with an explicit GPL-2.0 "
            "grant and multi-syllable entries, which is what a Maximum Matching tokenizer "
            "needs.",
            "It is pinned to one commit and the snapshot SHA256 is verified on every build, "
            "so the dictionary is reproducible and fully offline.",
        ],
        "offline_use": "engine.tokenizer loads resources/dict.txt; no network access at runtime.",
    }
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=SOURCE_DIR / RAW_NAME)
    parser.add_argument("--license", type=Path, default=SOURCE_DIR / LICENSE_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--fetch", action="store_true",
                        help="re-download the pinned commit before cleaning")
    args = parser.parse_args(argv)

    if args.fetch:
        print(f"fetching {UPSTREAM_RAW_URL}")
        fetch_pinned(UPSTREAM_RAW_URL, EXPECTED_RAW_SHA256, args.raw)
    if not args.raw.is_file():
        print(f"error: missing dictionary snapshot {args.raw}", file=sys.stderr)
        return 2

    actual = sha256_file(args.raw)
    pinned = args.raw.resolve() == (SOURCE_DIR / RAW_NAME).resolve()
    if not pinned and any(path.resolve() == default.resolve() for path, default in
                          ((args.output, DEFAULT_OUTPUT), (args.stats, DEFAULT_STATS),
                           (args.provenance, DEFAULT_PROVENANCE))):
        raise SourceVerificationError("Custom source requires separate --output, --stats and --provenance paths")
    verify_snapshot(args.raw, args.license, pinned)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    write_provenance(args.provenance, args.raw, args.license, pinned)
    stats = clean_dictionary(args.raw, args.output, args.stats, args.provenance)
    print(f"snapshot sha256 : {actual}")
    print(f"dictionary      : {stats['total_words']} entries -> {args.output}")
    print(f"output sha256   : {stats['output']['sha256']}")
    print(f"syllable spread : {stats['syllable_distribution']}")
    print(f"<=4 syllables   : {stats['share_within_4_syllables_percent']}%")
    print(f"discarded       : {stats['rejected_or_removed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
