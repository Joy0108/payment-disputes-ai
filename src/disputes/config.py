"""Configuration. Split boundaries and gate thresholds are policy, so they live here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DISPUTES_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
REGULATION_DIR = DATA_DIR / "regulations"
GOLDEN_PATH = DATA_DIR / "golden" / "rag_golden.json"
ARTIFACT_DIR = Path(os.environ.get("DISPUTES_ARTIFACT_DIR", ROOT / "artifacts"))
REPORT_DIR = Path(os.environ.get("DISPUTES_REPORT_DIR", ROOT / "reports"))

# Frozen up front, before any model was trained. Both boundaries are dates, not
# fractions: a fraction moves every time a row is added, and a moving boundary
# makes two runs incomparable and hides regressions as noise.
SPLIT_DATES = ("2023-04-01", "2023-11-01")
FRAUD_SPLIT_MONTHS = (5, 6)

COMPLAINT_ISSUES = [
    "Unauthorized transactions or other transaction problem",
    "Problem with a purchase shown on your statement",
    "Fraud or scam",
    "Incorrect information on your report",
    "Problem with a company's investigation into an existing problem",
    "Attempts to collect debt not owed",
    "Written notification about debt",
    "Managing an account",
    "Closing an account",
    "Problem caused by your funds being low",
    "Other features, terms, or problems",
    "Getting a credit card",
    "Trouble using your card",
    "Problem with a lender or other company charging your account",
    "Improper use of your report",
    "Communication tactics",
    "Struggling to repay your loan",
    "Charged fees or interest you didn't expect",
    "Money was not available when promised",
    "Confusing or missing disclosures",
]

RESPONSE_CATEGORIES = [
    "Closed with explanation",
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed without relief",
    "Untimely response",
]

FRAUD_FEATURES = [
    "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
    "customer_age", "days_since_request", "intended_balcon_amount", "zip_count_4w",
    "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "credit_risk_score", "email_is_free",
    "phone_home_valid", "phone_mobile_valid", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "session_length_in_minutes", "device_distinct_emails_8w",
    "device_fraud_count", "keep_alive_session", "bank_months_count",
]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    search_trials: int = 24
    random_state: int = 17
    # Fraud runs at a ~1% base rate. Accuracy is useless there and so is a
    # naive F1: the operating point is chosen by recall at a fixed alert budget,
    # because the review team's capacity is the real constraint.
    fraud_alert_budget: float = 0.02


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 4
    candidate_k: int = 20
    rrf_k: int = 60
    rerank: bool = True
    embed_dim: int = 96
    require_citations: bool = True


@dataclass(frozen=True)
class LlmConfig:
    backend: str = os.environ.get("DISPUTES_LLM", "template")  # template | anthropic
    anthropic_model: str = "claude-opus-5"
    prompt_version: str = "v3"


@dataclass(frozen=True)
class Gates:
    """Promotion gates. Each threshold is a decision with a consequence."""

    # Set below the numbers the committed report records, with enough margin
    # that ordinary run-to-run variation does not fail CI but a real regression
    # does. Two of them are far below what a textbook would call good, and that
    # is the honest level: outcome prediction against a drifting response policy
    # and fraud at a 1% base rate are hard, and a gate set at an aspiration is a
    # gate that gets disabled the first time it fires.
    issue_macro_f1: float = 0.78
    outcome_balanced_accuracy: float = 0.25
    escalation_roc_auc: float = 0.66
    fraud_recall_at_budget: float = 0.10
    rag_recall_at_k: float = 0.80
    citation_resolution: float = 0.99
    deadline_exactness: float = 1.0  # absolute: a deadline is right or it is a violation
    thresholds: dict[str, str] = field(default_factory=lambda: {
        "deadline_exactness": "Absolute. A miscomputed regulatory deadline is a violation, not a lower score.",
        "citation_resolution": "A citation that resolves to nothing is a fabricated reference in a consumer-facing letter.",
        "fraud_recall_at_budget": "Recall measured at the alert volume the review team can actually work.",
    })


DEFAULT_MODEL = ModelConfig(name="disputes")
DEFAULT_RAG = RagConfig()
DEFAULT_LLM = LlmConfig()
DEFAULT_GATES = Gates()


def ensure_dirs() -> None:
    for path in (ARTIFACT_DIR, REPORT_DIR, SILVER_DIR, GOLD_DIR):
        path.mkdir(parents=True, exist_ok=True)
