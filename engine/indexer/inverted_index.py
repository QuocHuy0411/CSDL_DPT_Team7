"""Inverted index over the L2-normalised TF-IDF CSR matrix.

Layout (CSR-like over the *term* axis, so a posting list is an O(1) slice)::

    postings.offsets: int64[n_features + 1]   boundaries into the two arrays below
    postings.docs:    int32[nnz]              document row indices, ascending per term
    postings.weights: float64[nnz]            L2-normalised TF-IDF weight of that pair

Because the stored weights are already row-normalised and the query vector is
normalised the same way, the cosine similarity of a candidate is simply
``sum(q_weight * d_weight)`` accumulated over the posting lists of the query
terms. Terms with an empty posting list (stopword-removed or OOV columns) are
represented by equal offsets and never enter the candidate set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .tfidf_engine import QueryVector, SCHEMA_VERSION, sha256_file

INDEX_FILE = "inverted_index.npz"


class InvertedIndex:
    """Term -> posting list mapping used to enumerate candidates and score them."""

    def __init__(self, vocabulary: dict[str, int] | None = None) -> None:
        self.vocabulary = dict(vocabulary) if vocabulary else None
        self.n_documents: int = 0
        self.n_features: int = 0
        self.offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self.docs: np.ndarray = np.zeros(0, dtype=np.int32)
        self.weights: np.ndarray = np.zeros(0, dtype=np.float64)
        self._fitted = False

    # ------------------------------------------------------------------ build

    def fit(self, matrix: csr_matrix, vocabulary: dict[str, int] | None = None) -> "InvertedIndex":
        """Build posting lists from a CSR matrix whose rows are L2-normalised."""
        matrix = matrix.tocsr().astype(np.float64)
        matrix.sort_indices()
        matrix.eliminate_zeros()
        if vocabulary is not None:
            self.vocabulary = dict(vocabulary)
        self.n_documents, self.n_features = matrix.shape
        row_of_entry = np.repeat(np.arange(self.n_documents, dtype=np.int32),
                                 np.diff(matrix.indptr).astype(np.int64))
        order = np.argsort(matrix.indices, kind="stable")
        self.docs = row_of_entry[order]
        self.weights = matrix.data[order]
        term_of_entry = matrix.indices[order]
        self.offsets = np.searchsorted(
            term_of_entry, np.arange(self.n_features + 1, dtype=np.int32), side="left"
        ).astype(np.int64)
        self.offsets[-1] = term_of_entry.size
        self._fitted = True
        return self

    # ------------------------------------------------------------------ query

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("InvertedIndex.fit must be called before this operation")

    def _term_id(self, term: int | str) -> int:
        if isinstance(term, str):
            if self.vocabulary is None:
                raise KeyError("This index has no vocabulary; pass a term id instead")
            position = self.vocabulary.get(term)
            if position is None:
                raise KeyError(f"Unknown term: {term}")
            return int(position)
        position = int(term)
        if not 0 <= position < self.n_features:
            raise KeyError(f"Term id out of range: {term}")
        return position

    def posting_slice(self, term: int | str) -> tuple[np.ndarray, np.ndarray]:
        """Document rows and their weights for one term, ascending by row."""
        self._require_fitted()
        position = self._term_id(term)
        start, end = int(self.offsets[position]), int(self.offsets[position + 1])
        return self.docs[start:end], self.weights[start:end]

    def posting_list(self, term: int | str) -> tuple[int, ...]:
        """Document rows containing the term (this is the classic posting list)."""
        docs, _ = self.posting_slice(term)
        return tuple(int(row) for row in docs)

    def posting_length(self, term: int | str) -> int:
        self._require_fitted()
        position = self._term_id(term)
        return int(self.offsets[position + 1] - self.offsets[position])

    def candidates(self, term_ids) -> np.ndarray:
        """Sorted unique union of the posting lists of the given term ids."""
        self._require_fitted()
        parts = [self.docs[int(self.offsets[position]):int(self.offsets[position + 1])]
                 for position in term_ids]
        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts)).astype(np.int32, copy=False)

    def score(self, query: QueryVector) -> tuple[np.ndarray, np.ndarray]:
        """Exact cosine scores of the candidate documents.

        Returns ``(document_rows, scores)`` aligned element-wise. Only rows
        present in at least one posting list of the query are touched; every
        stored weight is strictly positive, so every candidate score is > 0.
        """
        self._require_fitted()
        candidate_rows = self.candidates(query.term_ids)
        if candidate_rows.size == 0:
            return candidate_rows, np.zeros(0, dtype=np.float64)
        scores = np.zeros(candidate_rows.size, dtype=np.float64)
        for term_id, weight in zip(query.term_ids, query.weights):
            docs, values = self.posting_slice(int(term_id))
            if docs.size == 0:
                continue
            positions = np.searchsorted(candidate_rows, docs)
            scores += np.bincount(positions, weights=weight * values,
                                  minlength=candidate_rows.size)
        return candidate_rows, scores

    def cosine_scores_full(self, query: QueryVector) -> np.ndarray:
        """Scores for *every* document row (zeros outside the candidate set)."""
        self._require_fitted()
        scores = np.zeros(self.n_documents, dtype=np.float64)
        rows, values = self.score(query)
        scores[rows] = values
        return scores

    # ------------------------------------------------------------- inspection

    def __len__(self) -> int:
        return int(self.docs.size)

    def __contains__(self, term: object) -> bool:
        if isinstance(term, str) and self.vocabulary is not None:
            term = self.vocabulary.get(term)
        try:
            position = self._term_id(term)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            return False
        return int(self.offsets[position + 1]) > int(self.offsets[position])

    def stats(self) -> dict:
        self._require_fitted()
        lengths = np.diff(self.offsets)
        return {
            "n_documents": self.n_documents,
            "n_features": self.n_features,
            "total_postings": int(lengths.sum()),
            "empty_postings": int((lengths == 0).sum()),
            "max_posting_length": int(lengths.max()) if lengths.size else 0,
            "mean_posting_length": float(lengths.mean()) if lengths.size else 0.0,
            "bytes_docs": int(self.docs.nbytes),
            "bytes_weights": int(self.weights.nbytes),
        }

    # ----------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> str:
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / INDEX_FILE, docs=self.docs, offsets=self.offsets,
                 weights=self.weights,
                 n_documents=np.asarray(self.n_documents, dtype=np.int64),
                 n_features=np.asarray(self.n_features, dtype=np.int64),
                 schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64))
        return sha256_file(directory / INDEX_FILE)

    @classmethod
    def load(cls, directory: str | Path,
             vocabulary: dict[str, int] | None = None) -> "InvertedIndex":
        directory = Path(directory)
        with np.load(directory / INDEX_FILE) as payload:
            index = cls(vocabulary)
            index.docs = payload["docs"].astype(np.int32, copy=False)
            index.offsets = payload["offsets"].astype(np.int64, copy=False)
            index.weights = payload["weights"].astype(np.float64, copy=False)
            index.n_documents = int(payload["n_documents"])
            index.n_features = int(payload["n_features"])
            schema_version = int(payload["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported inverted index schema version: {schema_version}")
        if index.offsets.shape != (index.n_features + 1,):
            raise ValueError("Posting offsets do not match the stored vocabulary size")
        if int(index.offsets[-1]) != index.docs.size or index.docs.size != index.weights.size:
            raise ValueError("Posting arrays are inconsistent")
        index._fitted = True
        return index

    def __repr__(self) -> str:
        return (f"InvertedIndex(n_documents={self.n_documents}, n_features={self.n_features}, "
                f"postings={len(self)})")
