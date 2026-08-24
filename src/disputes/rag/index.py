"""Structure-aware retrieval over the regulation corpus.

Chunking is by *section*, not by token window, and that is the whole design.
A citation in a consumer-facing dispute letter has to name a provision the
recipient can look up - "12 CFR 1005.11(c)(2)" - and a sliding window that cuts
across 1005.11(c)(1) and (c)(2) produces a chunk that cites neither and answers
with a deadline drawn from both.

Retrieval is BM25 fused with a corpus-fitted dense projection. The lexical half
carries most of the weight here because regulatory queries are dense with exact
strings: section numbers, "ten business days", "sixty days", "provisional
credit". Those are precisely the tokens a low-dimensional embedding blurs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_RAG, REGULATION_DIR, RagConfig

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)*")
_STOP = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the to was were will with this
    these those which what when where how why does do can could should would must may shall""".split()
)

# Regulations spell numbers out and consumers do not. The threshold and the
# period are the most discriminative tokens in this corpus, so both spellings
# have to reach the same section. Substitution runs on the phrase before
# tokenisation - per token it fails, because "twenty-five" has already been
# split into two words by then.
_NUMBER_PHRASES = [
    (re.compile(r"\bsixty\b"), "sixty 60"),
    (re.compile(r"\b60\b"), "60 sixty"),
    (re.compile(r"\bforty[-\s]?five\b"), "forty-five 45"),
    (re.compile(r"\b45\b"), "45 forty-five"),
    (re.compile(r"\bthirty\b"), "thirty 30"),
    (re.compile(r"\b30\b"), "30 thirty"),
    (re.compile(r"\btwenty\b"), "twenty 20"),
    (re.compile(r"\b20\b"), "20 twenty"),
    (re.compile(r"\bninety\b"), "ninety 90"),
    (re.compile(r"\b90\b"), "90 ninety"),
    (re.compile(r"\bten\b"), "ten 10"),
    (re.compile(r"\b10\b"), "10 ten"),
    (re.compile(r"\bfive\b"), "five 5"),
    (re.compile(r"\bthree\b"), "three 3"),
    (re.compile(r"\btwo\b"), "two 2"),
]

# What a consumer or an intake agent calls a thing, against what the regulation
# calls it. Each entry is a term of art, not a general synonym.
_SYNONYMS = {
    "chargeback": ["billing", "error", "dispute"],
    "refund": ["credit", "correct"],
    "reversed": ["credit", "correct"],
    "temporary": ["provisional"],
    "stolen": ["unauthorized"],
    "hacked": ["unauthorized"],
    "skimmed": ["unauthorized"],
    "deadline": ["days", "period", "within"],
    "timeframe": ["days", "period", "within"],
    "cap": ["liability", "exceed"],
    "limit": ["liability", "exceed"],
    "harassment": ["harass", "abuse", "oppress"],
    "robocall": ["telephone", "repeatedly"],
    "validation": ["verification", "verify"],
    "tradeline": ["item", "information", "file"],
    "reinvestigate": ["reinvestigation"],
    "furnisher": ["furnisher", "person", "provided"],
    "atm": ["electronic", "fund", "transfer"],
    "ach": ["preauthorized", "electronic", "fund", "transfer"],
    "autopay": ["preauthorized"],
    "subscription": ["preauthorized", "recurring"],
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    for pattern, replacement in _NUMBER_PHRASES:
        text = pattern.sub(replacement, text)
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        if raw in _STOP:
            continue
        out.append(raw)
        out.extend(_SYNONYMS.get(raw, ()))
        # A section identifier is also indexed by its parts, so "1005.11" finds
        # "1005.11(c)(2)" and a query naming the part finds the whole.
        if "." in raw and raw[0].isdigit():
            out.extend(p for p in raw.split(".") if p)
    return out


@dataclass
class Section:
    id: str
    regulation: str
    title: str
    text: str

    @property
    def indexed_text(self) -> str:
        return f"{self.id} {self.regulation} {self.title}. {self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "regulation": self.regulation, "title": self.title}


def load_sections(directory: Path = REGULATION_DIR) -> list[Section]:
    path = directory / "sections.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Section(s["id"], s["regulation"], s["title"], s["text"]) for s in payload["sections"]]


class BM25:
    def __init__(self, sections: Sequence[Section], k1: float = 1.2, b: float = 0.7):
        self.k1, self.b = k1, b
        self.ids = [s.id for s in sections]
        self.freqs = [Counter(tokenize(s.indexed_text)) for s in sections]
        self.lengths = [sum(f.values()) for f in self.freqs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.df: Counter = Counter()
        for f in self.freqs:
            self.df.update(f.keys())

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            if idf <= 0:
                continue
            for i, freq in enumerate(self.freqs):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (self.lengths[i] / (self.avgdl or 1)))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]]))
        return [(self.ids[i], round(s, 6)) for i, s in ranked[:top_k]]


class DenseIndex:
    def __init__(self, sections: Sequence[Section], dim: int = 96):
        self.ids = [s.id for s in sections]
        docs = [tokenize(s.indexed_text) for s in sections]
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = max(1, len(docs))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)])

        matrix = np.zeros((len(docs), len(self.vocab)))
        for row, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                matrix[row, self.vocab[term]] = 1.0 + math.log(count)
        matrix *= self.idf
        matrix = _l2(matrix)
        k = int(min(dim, min(matrix.shape) - 1)) or 1
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        self.components = vt[:k].T
        self.vectors = _l2(matrix @ self.components)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for term, count in Counter(tokenize(text)).items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = 1.0 + math.log(count)
        vec *= self.idf
        return _l2((vec @ self.components).reshape(1, -1))[0]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        sims = self.vectors @ self.embed(query)
        order = np.argsort(-sims)[:top_k]
        return [(self.ids[i], round(float(sims[i]), 6)) for i in order]


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def rrf(rankings: Sequence[Sequence[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _s) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class RegulationIndex:
    def __init__(self, sections: Sequence[Section] | None = None, cfg: RagConfig = DEFAULT_RAG):
        self.cfg = cfg
        self.sections = list(sections if sections is not None else load_sections())
        self.by_id = {s.id: s for s in self.sections}
        self.bm25 = BM25(self.sections)
        self.dense = DenseIndex(self.sections, cfg.embed_dim)

    def rerank(self, query: str, candidates: Sequence[str], regulation: str | None = None) -> list[tuple[str, float]]:
        q_terms = set(tokenize(query))
        out = []
        for rank, sid in enumerate(candidates):
            section = self.by_id[sid]
            terms = set(tokenize(section.indexed_text))
            covered = sum(self.bm25.idf(t) for t in q_terms & terms)
            total = sum(self.bm25.idf(t) for t in q_terms) or 1.0
            title_hit = len(q_terms & set(tokenize(section.title))) / (len(q_terms) or 1)
            # A dispute already resolved under one regulation should not be
            # answered out of another. The boost is not a hard filter, because
            # cross-references are real - a Reg E dispute on a credit card can
            # implicate Reg Z - but the governing statute goes first.
            # Weighted above the fusion rank on purpose. "Unauthorized use of a
            # card" retrieves the Reg Z credit-card liability section and the
            # Reg E debit section about equally well, and answering a debit
            # dispute out of Reg Z is wrong on the law, not merely off-topic.
            # Off-regulation sections are demoted rather than removed, because
            # cross-references are real.
            same_reg = 1.0 if regulation and section.regulation == regulation else 0.0
            cross_reference = 0.35 if regulation and section.regulation == "CIRCULAR" else 0.0
            out.append((sid, 3.0 / (1 + rank) + 1.5 * (covered / total) + 0.9 * title_hit
                        + 2.6 * same_reg + cross_reference))
        return sorted(out, key=lambda kv: (-kv[1], kv[0]))

    def search(self, query: str, top_k: int | None = None, regulation: str | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.cfg.top_k
        lexical = self.bm25.search(query, self.cfg.candidate_k)
        dense = self.dense.search(query, self.cfg.candidate_k)
        order = [sid for sid, _ in rrf([lexical, dense], self.cfg.rrf_k)]
        if self.cfg.rerank:
            order = [sid for sid, _ in self.rerank(query, order[: self.cfg.candidate_k], regulation)]
        return [{**self.by_id[sid].to_dict(), "text": self.by_id[sid].text} for sid in order[:top_k]]
