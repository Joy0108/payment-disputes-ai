"""Tree-structured Parzen Estimator for hyperparameter search.

Random search is a fine default and this is not much more code. TPE is used
because the search here is over a handful of parameters with a strongly peaked
objective, which is exactly where modelling p(x | y) beats sampling uniformly.

The method: keep the trials whose score falls in the best ``gamma`` quantile as
the *good* set and the rest as the *bad* set, fit a kernel density estimate over
each, and propose the candidate maximising l(x)/g(x). No surrogate GP, no
acquisition-function machinery — the ratio of two densities is the acquisition
function.

Deterministic given a seed, because a hyperparameter search that cannot be
reproduced is not a result, it is an anecdote.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dimension:
    name: str
    low: float
    high: float
    kind: str = "float"  # float | int | log | choice
    choices: Sequence[Any] | None = None

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice":
            return rng.choice(list(self.choices or []))
        if self.kind == "log":
            return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        return rng.uniform(self.low, self.high)

    def to_internal(self, value: Any) -> float:
        if self.kind == "choice":
            return float(list(self.choices or []).index(value))
        if self.kind == "log":
            return math.log(value)
        return float(value)

    def from_internal(self, value: float) -> Any:
        if self.kind == "choice":
            options = list(self.choices or [])
            return options[max(0, min(len(options) - 1, int(round(value))))]
        if self.kind == "log":
            return math.exp(value)
        if self.kind == "int":
            return int(round(value))
        return float(value)

    @property
    def internal_range(self) -> tuple[float, float]:
        if self.kind == "choice":
            return 0.0, float(len(list(self.choices or [])) - 1)
        if self.kind == "log":
            return math.log(self.low), math.log(self.high)
        return float(self.low), float(self.high)


@dataclass
class Trial:
    params: dict[str, Any]
    score: float
    index: int


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_score: float
    trials: list[Trial] = field(default_factory=list)

    def history(self) -> list[dict[str, Any]]:
        return [{"trial": t.index, "score": round(t.score, 5), **t.params} for t in self.trials]

    def to_dict(self) -> dict[str, Any]:
        improving = []
        best = -math.inf
        for t in self.trials:
            if t.score > best:
                best = t.score
                improving.append({"trial": t.index, "score": round(t.score, 5)})
        return {
            "trials": len(self.trials),
            "best_score": round(self.best_score, 5),
            "best_params": self.best_params,
            "improvements": improving,
        }


def tpe_search(
    objective: Callable[[dict[str, Any]], float],
    space: Sequence[Dimension],
    n_trials: int = 24,
    n_startup: int = 8,
    gamma: float = 0.25,
    n_candidates: int = 24,
    seed: int = 17,
) -> SearchResult:
    rng = random.Random(seed)
    trials: list[Trial] = []

    for index in range(n_trials):
        if index < n_startup or len(trials) < 4:
            # Random start. TPE needs observations before its densities mean
            # anything, and seeding it with too few makes it exploit noise.
            params = {d.name: d.sample(rng) for d in space}
        else:
            params = _propose(trials, space, rng, gamma, n_candidates)
        score = objective(params)
        trials.append(Trial(params=params, score=float(score), index=index))

    best = max(trials, key=lambda t: t.score)
    return SearchResult(best_params=best.params, best_score=best.score, trials=trials)


def _propose(
    trials: list[Trial], space: Sequence[Dimension], rng: random.Random, gamma: float, n_candidates: int
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda t: -t.score)
    n_good = max(2, int(math.ceil(gamma * len(ordered))))
    good, bad = ordered[:n_good], ordered[n_good:] or ordered[-2:]

    candidates = []
    for _ in range(n_candidates):
        candidate: dict[str, Any] = {}
        for dim in space:
            # Sample around a randomly chosen good observation, with a bandwidth
            # that shrinks as observations accumulate.
            anchor = rng.choice(good)
            lo, hi = dim.internal_range
            bandwidth = max((hi - lo) / max(4.0, math.sqrt(len(good) * 4)), 1e-6)
            value = rng.gauss(dim.to_internal(anchor.params[dim.name]), bandwidth)
            candidate[dim.name] = dim.from_internal(max(lo, min(hi, value)))
        candidates.append(candidate)

    def ratio(candidate: dict[str, Any]) -> float:
        return _log_density(candidate, good, space) - _log_density(candidate, bad, space)

    return max(candidates, key=ratio)


def _log_density(candidate: dict[str, Any], observations: list[Trial], space: Sequence[Dimension]) -> float:
    """Log of a product of one-dimensional Parzen estimates.

    Independence across dimensions is an approximation, and the one TPE makes.
    It is why TPE handles ten parameters comfortably and does not model the
    interaction between two of them.
    """
    total = 0.0
    for dim in space:
        lo, hi = dim.internal_range
        bandwidth = max((hi - lo) / max(4.0, math.sqrt(len(observations) * 4)), 1e-6)
        x = dim.to_internal(candidate[dim.name])
        density = 0.0
        for obs in observations:
            mu = dim.to_internal(obs.params[dim.name])
            density += math.exp(-0.5 * ((x - mu) / bandwidth) ** 2) / (bandwidth * math.sqrt(2 * math.pi))
        total += math.log(max(density / len(observations), 1e-12))
    return total


def random_search(
    objective: Callable[[dict[str, Any]], float],
    space: Sequence[Dimension],
    n_trials: int = 24,
    seed: int = 17,
) -> SearchResult:
    """Baseline, for the comparison that justifies using TPE at all."""
    rng = random.Random(seed)
    trials = []
    for index in range(n_trials):
        params = {d.name: d.sample(rng) for d in space}
        trials.append(Trial(params=params, score=float(objective(params)), index=index))
    best = max(trials, key=lambda t: t.score)
    return SearchResult(best_params=best.params, best_score=best.score, trials=trials)
