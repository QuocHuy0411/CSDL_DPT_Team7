"""Text normalization for the M1 Vietnamese tokenizer.

`normalize` returns the display form required by the API contract: NFC,
lowercase, URLs/emails/HTML and digits removed, everything that is not a
Vietnamese letter turned into a space, and whitespace collapsed to one space.

Punctuation is a boundary, not noise. `normalize` shows the collapsed text, but a
word must never be fused across a punctuation mark, so the tokenizer consumes
`normalize_segments`, which keeps the boundary information that the display form
throws away:

    normalize_segments("tôi, đi!")      -> ["tôi", "đi"]
    normalize("tôi, đi!")               -> "tôi đi"

The two agree on the letters; only the second one is allowed to look inside.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "VIETNAMESE_LETTERS",
    "VIETNAMESE_LETTERS_SET",
    "normalize",
    "normalize_segments",
    "syllables",
    "policy",
]

# The Vietnamese alphabet: 29 base letters plus every tone mark the language uses.
BASE = "aăâbcdđeêghiklmnoôơpqrstuưvxy"
TONES = ("áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
         "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụưứừửữựýỳỷỹỵ")
# Correction to the narrow sample regex in the project plan: that pattern dropped
# every ASCII letter, which silently destroyed loanwords and proper names that are
# genuinely written in Vietnamese text. "Windows" became "indows", "Fed" became
# "ed", "Java" became "ava". These four Latin letters are the discriminative ones in
# such names (w for Windows/www, j for Java, f for Fed, z for names such as "Zalo"),
# so they are kept. They are NOT added to the dictionary: they stay out-of-vocabulary
# single tokens, which is the honest outcome, instead of being mangled.
LOANWORD_LATIN = "fjwz"
# 'ư' occurs in both BASE and TONES, so the alphabet is de-duplicated here.
VIETNAMESE_LETTERS = "".join(dict.fromkeys(LOANWORD_LATIN + BASE + TONES))
VIETNAMESE_LETTERS_SET = frozenset(VIETNAMESE_LETTERS)

# Only well formed tags are removed. A bare "<" in prose is left to the letter
# filter, which turns it into a space instead of swallowing the rest of a line.
HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
URL = re.compile(r"(?:https?://|ftp://|www\.)\S+")
# The local part must start at a boundary. Without the lookbehind, the ASCII run
# inside a Vietnamese word is swallowed: "họca@b.com" matched "ca@b.com" and left
# "họ" behind. With it, the address is simply not recognised when it is glued to a
# word, and the '@' still acts as a boundary, which is the safer outcome.
EMAIL = re.compile(rf"(?<![{VIETNAMESE_LETTERS}])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{{2,}}")
NON_LETTER = re.compile(rf"[^{VIETNAMESE_LETTERS}\s]")
BOM = "﻿"
#: Removed spans (URL, email, HTML tag) are replaced with a hard boundary, not a
#: space. A space is a syllable separator inside a Vietnamese word, so substituting
#: one for a removed span would silently glue the text on either side of it into a
#: single segment: "học<br> sinh" would segment as one word "học sinh". A newline is
#: already a boundary in `normalize_segments`, so removal is recorded as one.
BOUNDARY = "\n"


def _strip_non_text(text: str) -> str:
    text = HTML_TAG.sub(BOUNDARY, text)
    text = URL.sub(BOUNDARY, text)
    text = EMAIL.sub(BOUNDARY, text)
    return text


def normalize_segments(text: str) -> list[str]:
    """Normalized text split at every punctuation boundary.

    A segment is a maximal run of Vietnamese letters **and the plain spaces
    between them**, because in Vietnamese a space separates syllables inside a
    word, not words from each other:

        normalize_segments("cơ sở dữ liệu")  -> ["cơ sở dữ liệu"]
        normalize_segments("tôi, đi!")        -> ["tôi", "đi"]
        normalize_segments("câu 1. tiếp")     -> ["câu", "tiếp"]
        normalize_segments("một\\ntư")        -> ["một", "tư"]

    Punctuation, digits, markup, URLs and any whitespace that is not a plain
    space (newline, tab) all end a segment. A tokenizer built on these segments
    therefore cannot fuse a word across punctuation or across a line break.
    """
    if not isinstance(text, str) or not text:
        return []
    text = unicodedata.normalize("NFC", text.replace(BOM, "")).lower()
    text = _strip_non_text(text)
    segments: list[str] = []
    buffer: list[str] = []
    pending_space = False
    for character in text:
        if character in VIETNAMESE_LETTERS_SET:
            if pending_space and buffer:
                buffer.append(" ")
            pending_space = False
            buffer.append(character)
        elif character == " ":
            # an ordinary space is a syllable separator, not a boundary
            pending_space = True
        else:
            # punctuation, digits, newline, tab, anything else: hard boundary
            pending_space = False
            if buffer:
                segments.append("".join(buffer))
                buffer = []
    if buffer:
        segments.append("".join(buffer))
    return segments


def normalize(text: str) -> str:
    """Contract display form: boundaries collapsed to a single space."""
    return " ".join(normalize_segments(text))


def syllables(text: str) -> list[str]:
    """All normalized syllables of `text`, in order, ignoring boundaries.

    Provided for inspection and tests. The tokenizer itself works on segments so
    that boundaries survive.
    """
    syllables: list[str] = []
    for segment in normalize_segments(text):
        syllables.extend(segment.split(" "))
    return syllables


def policy() -> dict:
    """Machine readable description of the normalization policy."""
    return {
        "schema_version": 1,
        "order": [
            "strip BOM",
            "unicode NFC",
            "lowercase (str.lower)",
            "remove HTML tags, URLs, emails",
            "keep Vietnamese letters and spaces; every other character ends a segment",
        ],
        "punctuation_is_a_boundary": True,
        "removed_spans_are_boundaries": True,
        "removed_span_note": "URL, email and HTML tag spans are replaced with a hard "
                             "boundary, never with a space, so the text on either side "
                             "cannot be glued into one word",
        "whitespace_is_a_boundary": False,
        "whitespace_note": "a space separates syllables inside a Vietnamese word, so a "
                          "space alone never ends a segment",
        "digits": "end a segment, like any other non-letter",
        "alphabet_size": len(VIETNAMESE_LETTERS),
        "latin_letters_kept": LOANWORD_LATIN,
        "latin_letter_note": (
            "the sample regex in the project plan removed every ASCII letter and turned "
            "Windows/Fed/Java/www into indows/ed/ava/ww. f, j, w and z are kept so "
            "loanwords survive as out-of-vocabulary tokens instead of being mangled. "
            "The dictionary itself stays Vietnamese-only; these letters are not added to it."
        ),
        "dictionary_scope": "resources/dict.txt contains Vietnamese entries only",
        "display_form": "normalize() collapses boundaries, normalize_segments() keeps them",
        "note": "normalize() output alone is NOT sufficient input for a tokenizer; "
                "normalization must not be allowed to hide punctuation boundaries.",
    }