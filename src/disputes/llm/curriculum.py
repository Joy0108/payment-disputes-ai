"""Does curriculum ordering actually help, on a model this project can train?

The fine-tune itself is a QLoRA run on a 7B model (``qlora.py``) and is not
something CI can execute. But the *claim* underneath a curriculum - that showing
easy examples before hard ones changes what a model ends up with - is testable
on any model trained by stochastic gradient descent over an ordered stream, and
this project has one: the issue classifier.

So the experiment runs there. A linear classifier is trained with plain SGD over
a single pass in three orderings - easy-to-hard, shuffled, and hard-to-easy -
with everything else held fixed and the same seed. One pass matters: with enough
epochs the ordering washes out, and the regime where a curriculum can help is
exactly the one where the data is seen once, which is also the regime a QLoRA
fine-tune on a few thousand examples is in.

The result is reported as measured, including if the curriculum loses.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CurriculumResult:
    ordering: str
    accuracy: float
    macro_f1: float
    final_loss: float
    loss_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordering": self.ordering,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "final_loss": round(self.final_loss, 4),
        }


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _train_sgd(
    X: np.ndarray, y: np.ndarray, order: Sequence[int], n_classes: int, lr: float = 0.35, seed: int = 0
) -> tuple[np.ndarray, list[float]]:
    """One pass of multinomial SGD over the given ordering."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(X.shape[1], n_classes))
    losses: list[float] = []
    running = 0.0
    for step, idx in enumerate(order, start=1):
        x = X[idx]
        probs = _softmax(x @ W)
        target = np.zeros(n_classes)
        target[y[idx]] = 1.0
        running += -np.log(max(probs[y[idx]], 1e-12))
        W -= lr * np.outer(x, probs - target)
        if step % max(1, len(order) // 20) == 0:
            losses.append(running / step)
    return W, losses


def _metrics(W: np.ndarray, X: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[float, float]:
    predicted = np.argmax(X @ W, axis=1)
    accuracy = float((predicted == y).mean())
    f1s = []
    for c in range(n_classes):
        tp = int(((predicted == c) & (y == c)).sum())
        fp = int(((predicted == c) & (y != c)).sum())
        fn = int(((predicted != c) & (y == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return accuracy, float(np.mean(f1s))


def run_experiment(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    difficulty: dict[str, float],
    label_key: str = "label_issue",
    max_features: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Three orderings, one pass each, everything else identical."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(max_features=max_features, sublinear_tf=True, min_df=2, strip_accents="unicode")
    Xtr = vectoriser.fit_transform([r["narrative"] for r in train_rows]).toarray()
    Xte = vectoriser.transform([r["narrative"] for r in test_rows]).toarray()

    classes = sorted({r[label_key] for r in train_rows})
    index = {c: i for i, c in enumerate(classes)}
    ytr = np.array([index[r[label_key]] for r in train_rows])
    yte = np.array([index.get(r[label_key], 0) for r in test_rows])

    scores = [difficulty.get(r["complaint_id"], 0.5) for r in train_rows]
    easy_first = list(np.argsort(scores))
    hard_first = list(reversed(easy_first))
    shuffled = list(range(len(train_rows)))
    random.Random(seed).shuffle(shuffled)

    results = []
    for name, order in [("easy_to_hard", easy_first), ("shuffled", shuffled), ("hard_to_easy", hard_first)]:
        W, losses = _train_sgd(Xtr, ytr, order, len(classes), seed=seed)
        accuracy, macro_f1 = _metrics(W, Xte, yte, len(classes))
        results.append(CurriculumResult(name, accuracy, macro_f1, losses[-1] if losses else float("nan"), losses))

    by_name = {r.ordering: r for r in results}
    delta = by_name["easy_to_hard"].macro_f1 - by_name["shuffled"].macro_f1
    leakage = difficulty_label_coupling(train_rows, scores, label_key)

    return {
        "protocol": "single pass of multinomial SGD, identical seed, identical features, ordering is the only difference",
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "classes": len(classes),
        "results": [r.to_dict() for r in results],
        "curriculum_delta_macro_f1": round(delta, 4),
        "difficulty_label_coupling": leakage,
        "verdict": (
            "curriculum ordering helps on a single pass"
            if delta > 0.005 else
            "curriculum ordering hurts on a single pass"
            if delta < -0.005 else
            "no measurable difference from ordering at this scale"
        ),
        "diagnosis": (
            f"Difficulty is not independent of the label: {leakage['eta_squared']:.0%} of its variance is explained "
            "by the class. Sorting by difficulty therefore partially sorts by class, and a single pass of SGD over "
            "class-blocked data ends fitted to whatever it saw last. The ordering is not teaching the model an "
            "easier version of the problem first; it is removing the shuffling that stops recency bias."
            if leakage["eta_squared"] > 0.15 else
            "Difficulty is close to independent of the label, so the ordering effect is not a class-blocking artefact."
        ),
        "caveat": (
            "One pass over a linear model is the regime where ordering can matter. It is evidence about the "
            "curriculum, not about the QLoRA fine-tune, which is not run here."
        ),
    }


def difficulty_label_coupling(
    rows: Sequence[dict[str, Any]], scores: Sequence[float], label_key: str
) -> dict[str, Any]:
    """How much of the difficulty signal is really the label.

    A curriculum assumes difficulty and label are independent. When they are
    not, ordering by difficulty groups the classes, and any effect measured is
    an artefact of that grouping rather than of the curriculum.
    """
    grand = float(np.mean(scores)) if scores else 0.0
    groups: dict[Any, list[float]] = {}
    for row, score in zip(rows, scores, strict=True):
        groups.setdefault(row[label_key], []).append(score)

    ss_between = sum(len(v) * (float(np.mean(v)) - grand) ** 2 for v in groups.values())
    ss_total = float(sum((s - grand) ** 2 for s in scores)) or 1.0
    return {
        "eta_squared": round(ss_between / ss_total, 4),
        "classes": len(groups),
        "note": "fraction of difficulty variance explained by the label; a curriculum assumes this is near zero",
    }
