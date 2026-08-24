"""The four models, with the metric each one is actually judged on.

Sharing a feature pipeline across the three complaint models is deliberate: the
narrative is the same text, and three separate vectorisers would mean three
vocabularies, three sets of preprocessing bugs, and a change to one that
silently does not reach the others.

Each model is scored on a metric chosen for the decision it supports, and none
of them is accuracy:

* **issue classification** - macro F1 across twenty classes, because the head of
  the distribution is easy and the tail is where routing goes wrong;
* **outcome prediction** - balanced accuracy across five imbalanced classes;
* **escalation risk** - ROC AUC, because it feeds a queue ordering rather than a
  hard decision;
* **fraud** - recall at a fixed alert budget. At a 1% base rate, F1 optimises an
  operating point nobody works; the review team can handle a fixed number of
  alerts a day and the only question is how much fraud is inside that number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ARTIFACT_DIR, DEFAULT_MODEL, FRAUD_FEATURES, ModelConfig
from ..data.layers import load_gold
from .search import Dimension, SearchResult, tpe_search


@dataclass
class TrainedModel:
    name: str
    target: str
    metric: str
    params: dict[str, Any]
    validation_score: float
    test_score: float
    extra: dict[str, Any] = field(default_factory=dict)
    search: SearchResult | None = None
    estimator: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "target": self.target,
            "metric": self.metric,
            "validation_score": round(self.validation_score, 4),
            "test_score": round(self.test_score, 4),
            "best_params": self.params,
            "search": self.search.to_dict() if self.search else None,
            **self.extra,
        }


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        out[row["split"]].append(row)
    return out


def _text_pipeline(min_df: int, max_features: int, ngram_high: int, C: float, class_weight: str | None) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=int(min_df),
            max_features=int(max_features),
            ngram_range=(1, int(ngram_high)),
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(C=float(C), max_iter=2000, class_weight=class_weight, solver="lbfgs")),
    ])


TEXT_SPACE = [
    Dimension("C", 0.05, 40.0, "log"),
    Dimension("min_df", 1, 6, "int"),
    Dimension("max_features", 4000, 40000, "int"),
    Dimension("ngram_high", 1, 2, "int"),
    Dimension("class_weight", 0, 1, "choice", choices=[None, "balanced"]),
]


# ---------------------------------------------------------------------------
# 1. issue classification
# ---------------------------------------------------------------------------

def train_issue_classifier(splits: dict[str, list[dict]], cfg: ModelConfig = DEFAULT_MODEL) -> TrainedModel:
    Xtr = [r["narrative"] for r in splits["train"]]
    ytr = [r["label_issue"] for r in splits["train"]]
    Xva = [r["narrative"] for r in splits["validation"]]
    yva = [r["label_issue"] for r in splits["validation"]]
    Xte = [r["narrative"] for r in splits["test"]]
    yte = [r["label_issue"] for r in splits["test"]]

    def objective(params: dict[str, Any]) -> float:
        model = _text_pipeline(params["min_df"], params["max_features"], params["ngram_high"],
                               params["C"], params["class_weight"])
        model.fit(Xtr, ytr)
        return f1_score(yva, model.predict(Xva), average="macro", zero_division=0)

    result = tpe_search(objective, TEXT_SPACE, n_trials=cfg.search_trials, seed=cfg.random_state)
    best = _text_pipeline(result.best_params["min_df"], result.best_params["max_features"],
                          result.best_params["ngram_high"], result.best_params["C"],
                          result.best_params["class_weight"])
    best.fit(Xtr + Xva, ytr + yva)
    predicted = best.predict(Xte)

    per_class = f1_score(yte, predicted, average=None, labels=sorted(set(ytr)), zero_division=0)
    weakest = sorted(zip(sorted(set(ytr)), per_class, strict=True), key=lambda kv: kv[1])[:3]

    return TrainedModel(
        name="issue_classifier", target="label_issue", metric="macro_f1",
        params=result.best_params, validation_score=result.best_score,
        test_score=f1_score(yte, predicted, average="macro", zero_division=0),
        extra={
            "classes": len(set(ytr)),
            "test_micro_f1": round(f1_score(yte, predicted, average="micro", zero_division=0), 4),
            "weakest_classes": [{"issue": k, "f1": round(float(v), 4)} for k, v in weakest],
        },
        search=result, estimator=best,
    )


# ---------------------------------------------------------------------------
# 2. outcome prediction
# ---------------------------------------------------------------------------

def train_outcome_model(splits: dict[str, list[dict]], cfg: ModelConfig = DEFAULT_MODEL) -> TrainedModel:
    """Predict the company response category.

    The hard part is not the model. Company response behaviour *drifts* in this
    data, so a model fitted on 2020-2022 is predicting a policy that no longer
    applies by 2024. The temporal split is what makes that visible; a random
    split would report a materially better number for a model that is worse.
    """
    def featurise(rows):
        return [f"{r['narrative']} __issue_{r['issue'].replace(' ', '_')} __size_{r['company_size']} __via_{r['submitted_via']}"
                for r in rows]

    Xtr, ytr = featurise(splits["train"]), [r["label_outcome"] for r in splits["train"]]
    Xva, yva = featurise(splits["validation"]), [r["label_outcome"] for r in splits["validation"]]
    Xte, yte = featurise(splits["test"]), [r["label_outcome"] for r in splits["test"]]

    def objective(params):
        model = _text_pipeline(params["min_df"], params["max_features"], params["ngram_high"],
                               params["C"], params["class_weight"])
        model.fit(Xtr, ytr)
        return balanced_accuracy_score(yva, model.predict(Xva))

    result = tpe_search(objective, TEXT_SPACE, n_trials=cfg.search_trials, seed=cfg.random_state + 1)
    best = _text_pipeline(result.best_params["min_df"], result.best_params["max_features"],
                          result.best_params["ngram_high"], result.best_params["C"],
                          result.best_params["class_weight"])
    best.fit(Xtr + Xva, ytr + yva)
    predicted = best.predict(Xte)

    # What a random split would have reported, for the same model class.
    shuffled = _random_split_reference(featurise, splits, "label_outcome", result.best_params, balanced_accuracy_score)

    return TrainedModel(
        name="outcome_model", target="label_outcome", metric="balanced_accuracy",
        params=result.best_params, validation_score=result.best_score,
        test_score=balanced_accuracy_score(yte, predicted),
        extra={
            "classes": len(set(ytr)),
            "random_split_reference": round(shuffled, 4),
            "temporal_penalty": round(shuffled - balanced_accuracy_score(yte, predicted), 4),
            "note": (
                "random_split_reference is what a shuffled split would have reported for the same model. "
                "The gap is the response-policy drift the temporal split refuses to hide."
            ),
        },
        search=result, estimator=best,
    )


def _random_split_reference(featurise, splits, label, params, scorer) -> float:
    rng = np.random.default_rng(0)
    everything = splits["train"] + splits["validation"] + splits["test"]
    order = rng.permutation(len(everything))
    cut = int(len(everything) * 0.84)
    train_rows = [everything[i] for i in order[:cut]]
    test_rows = [everything[i] for i in order[cut:]]
    model = _text_pipeline(params["min_df"], params["max_features"], params["ngram_high"],
                           params["C"], params["class_weight"])
    model.fit(featurise(train_rows), [r[label] for r in train_rows])
    return float(scorer([r[label] for r in test_rows], model.predict(featurise(test_rows))))


# ---------------------------------------------------------------------------
# 3. escalation risk
# ---------------------------------------------------------------------------

def train_escalation_model(splits: dict[str, list[dict]], cfg: ModelConfig = DEFAULT_MODEL) -> TrainedModel:
    def featurise(rows):
        return [f"{r['narrative']} __issue_{r['issue'].replace(' ', '_')} __resp_{r['label_outcome'].replace(' ', '_')} "
                f"__size_{r['company_size']} __timely_{r['timely_response']}" for r in rows]

    Xtr, ytr = featurise(splits["train"]), [int(r["label_escalation"]) for r in splits["train"]]
    Xva, yva = featurise(splits["validation"]), [int(r["label_escalation"]) for r in splits["validation"]]
    Xte, yte = featurise(splits["test"]), [int(r["label_escalation"]) for r in splits["test"]]

    def objective(params):
        model = _text_pipeline(params["min_df"], params["max_features"], params["ngram_high"],
                               params["C"], params["class_weight"])
        model.fit(Xtr, ytr)
        return roc_auc_score(yva, model.predict_proba(Xva)[:, 1])

    result = tpe_search(objective, TEXT_SPACE, n_trials=cfg.search_trials, seed=cfg.random_state + 2)
    best = _text_pipeline(result.best_params["min_df"], result.best_params["max_features"],
                          result.best_params["ngram_high"], result.best_params["C"],
                          result.best_params["class_weight"])
    best.fit(Xtr + Xva, ytr + yva)
    scores = best.predict_proba(Xte)[:, 1]

    return TrainedModel(
        name="escalation_model", target="label_escalation", metric="roc_auc",
        params=result.best_params, validation_score=result.best_score,
        test_score=roc_auc_score(yte, scores),
        extra={"positive_rate": round(float(np.mean(yte)), 4)},
        search=result, estimator=best,
    )


# ---------------------------------------------------------------------------
# 4. fraud detection
# ---------------------------------------------------------------------------

def recall_at_budget(y_true, scores, budget: float) -> tuple[float, float]:
    """Recall when only the top ``budget`` fraction of applications is alerted."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = max(1, int(round(len(scores) * budget)))
    top = np.argsort(-scores)[:k]
    caught = int(y_true[top].sum())
    total = int(y_true.sum())
    precision = caught / k
    return (caught / total if total else float("nan")), precision


def train_fraud_model(splits: dict[str, list[dict]], cfg: ModelConfig = DEFAULT_MODEL) -> TrainedModel:
    def matrix(rows):
        return np.array([[float(r[f]) for f in FRAUD_FEATURES] for r in rows])

    Xtr, ytr = matrix(splits["train"]), np.array([r["fraud_bool"] for r in splits["train"]])
    Xva, yva = matrix(splits["validation"]), np.array([r["fraud_bool"] for r in splits["validation"]])
    Xte, yte = matrix(splits["test"]), np.array([r["fraud_bool"] for r in splits["test"]])

    space = [
        Dimension("C", 0.005, 20.0, "log"),
        Dimension("class_weight", 0, 1, "choice", choices=[None, "balanced"]),
    ]

    def build(params):
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=float(params["C"]), class_weight=params["class_weight"],
                                       max_iter=3000, random_state=cfg.random_state)),
        ])

    def objective(params):
        model = build(params)
        model.fit(Xtr, ytr)
        recall, _ = recall_at_budget(yva, model.predict_proba(Xva)[:, 1], cfg.fraud_alert_budget)
        return recall

    result = tpe_search(objective, space, n_trials=max(10, cfg.search_trials // 2), seed=cfg.random_state + 3)
    best = build(result.best_params)
    best.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    scores = best.predict_proba(Xte)[:, 1]
    recall, precision = recall_at_budget(yte, scores, cfg.fraud_alert_budget)

    # The concept drift is on keep_alive_session, whose coefficient sign flips
    # halfway through the window. Reporting the coefficient makes the effect
    # visible instead of leaving it as an unexplained metric drop.
    coefficients = dict(zip(FRAUD_FEATURES, best.named_steps["clf"].coef_[0], strict=True))

    return TrainedModel(
        name="fraud_model", target="fraud_bool", metric=f"recall@{cfg.fraud_alert_budget:.0%}_alert_budget",
        params=result.best_params, validation_score=result.best_score, test_score=recall,
        extra={
            "alert_budget": cfg.fraud_alert_budget,
            "precision_at_budget": round(precision, 4),
            "test_roc_auc": round(float(roc_auc_score(yte, scores)), 4),
            "base_rate": round(float(yte.mean()), 5),
            "lift_over_random": round(precision / float(yte.mean()), 2) if yte.mean() else None,
            "keep_alive_coefficient": round(float(coefficients["keep_alive_session"]), 4),
            "note": (
                "keep_alive_session inverts its relationship to the label partway through the window. A model "
                "fitted across the whole period averages the two regimes and its coefficient lands near zero, "
                "which is the shape a concept drift leaves behind."
            ),
        },
        search=result, estimator=best,
    )


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def train_all(cfg: ModelConfig = DEFAULT_MODEL, write: bool = True) -> dict[str, TrainedModel]:
    complaints = split_rows(load_gold("complaints"))
    fraud = split_rows(load_gold("fraud"))

    models = {
        "issue_classifier": train_issue_classifier(complaints, cfg),
        "outcome_model": train_outcome_model(complaints, cfg),
        "escalation_model": train_escalation_model(complaints, cfg),
        "fraud_model": train_fraud_model(fraud, cfg),
    }

    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "models.json").write_text(
            json.dumps({k: m.to_dict() for k, m in models.items()}, indent=2), encoding="utf-8", newline="\n")
        _persist(models)
    return models


def _persist(models: dict[str, TrainedModel]) -> Path | None:
    try:
        import joblib
    except ImportError:  # pragma: no cover - joblib ships with scikit-learn
        return None
    path = ARTIFACT_DIR / "estimators.joblib"
    joblib.dump({k: m.estimator for k, m in models.items()}, path)
    return path


def load_estimators() -> dict[str, Any]:
    import joblib

    path = ARTIFACT_DIR / "estimators.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `disputes train`")
    return joblib.load(path)
