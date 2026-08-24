"""Generate the three raw corpora.

Stand-ins for the real sources, built to carry the properties that matter for
modelling rather than to look like a plausible sample:

* **Complaints** - the shape of the CFPB Consumer Complaint Database. Issue
  taxonomy, company response categories, narrative text, and a submission date
  spanning four years. The generator deliberately reproduces three defects the
  real data has: class imbalance across issues, a *drift* in company response
  behaviour partway through the period, and narratives whose signal for the
  label is real but noisy.

* **Fraud** - the shape of the Feedzai Bank Account Fraud suite: a tabular
  application-fraud problem at a ~1.2% base rate with a month index for
  temporal splitting, correlated features, and one feature whose relationship
  to the label inverts halfway through, because that is what makes a temporal
  split matter rather than a formality.

* **Regulations** - written by hand in ``data/regulations``, not generated.

Ground truth for the drift experiments is written to ``data/raw/meta.json``.
Nothing outside evaluation may read it.

    python scripts/build_data.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

from narratives import compose

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 20240815

# csv.writer terminates rows with CR LF on every platform, not with the platform
# default. A file written here on Windows and one written on Linux therefore
# differ on every line while holding identical data, and CI compares them byte
# for byte to prove the generator is reproducible. Pin the terminator instead of
# inheriting it from the dialect.
CSV_DIALECT = {"lineterminator": "\n"}

N_COMPLAINTS = 9000
N_FRAUD = 24000
START = date(2020, 7, 1)
END = date(2024, 6, 30)

# --------------------------------------------------------------------------
# complaint taxonomy
# --------------------------------------------------------------------------
#
# Product / sub-product / issue follows the CFPB taxonomy. The issue label is
# what the classifier predicts, and the regulation column is what the rules
# engine keys deadline calculations off - so an issue mapped to the wrong
# regulation is a wrong deadline, not just a wrong label.

ISSUES = [
    # (issue, product, regulation, base_weight, escalation_prior)
    ("Unauthorized transactions or other transaction problem", "Checking or savings account", "REG_E", 0.11, 0.35),
    ("Problem with a purchase shown on your statement", "Credit card", "REG_Z", 0.10, 0.30),
    ("Fraud or scam", "Money transfer, virtual currency, or money service", "REG_E", 0.08, 0.55),
    ("Incorrect information on your report", "Credit reporting", "FCRA", 0.12, 0.25),
    ("Problem with a company's investigation into an existing problem", "Credit reporting", "FCRA", 0.09, 0.40),
    ("Attempts to collect debt not owed", "Debt collection", "FDCPA", 0.08, 0.45),
    ("Written notification about debt", "Debt collection", "FDCPA", 0.05, 0.20),
    ("Managing an account", "Checking or savings account", "REG_E", 0.06, 0.15),
    ("Closing an account", "Checking or savings account", "REG_E", 0.04, 0.25),
    ("Problem caused by your funds being low", "Checking or savings account", "REG_E", 0.04, 0.20),
    ("Other features, terms, or problems", "Credit card", "REG_Z", 0.04, 0.15),
    ("Getting a credit card", "Credit card", "REG_Z", 0.03, 0.10),
    ("Trouble using your card", "Prepaid card", "REG_E", 0.03, 0.25),
    ("Problem with a lender or other company charging your account", "Checking or savings account", "REG_E", 0.04, 0.35),
    ("Improper use of your report", "Credit reporting", "FCRA", 0.03, 0.30),
    ("Communication tactics", "Debt collection", "FDCPA", 0.03, 0.40),
    ("Struggling to repay your loan", "Payday loan, title loan, or personal loan", "REG_Z", 0.02, 0.20),
    ("Charged fees or interest you didn't expect", "Credit card", "REG_Z", 0.05, 0.25),
    ("Money was not available when promised", "Money transfer, virtual currency, or money service", "REG_E", 0.03, 0.35),
    ("Confusing or missing disclosures", "Checking or savings account", "REG_E", 0.03, 0.15),
]

RESPONSES = [
    "Closed with explanation",
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed without relief",
    "Untimely response",
]

CHANNELS = ["Web", "Phone", "Referral", "Postal mail", "Fax"]
STATES = ["CA", "TX", "FL", "NY", "GA", "IL", "PA", "OH", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA"]

# Large institutions are over-represented in the real database because they are
# the ones the CFPB routes complaints to and the ones consumers know how to
# complain about. Reproduced here so the model card can document it.
COMPANIES = (
    [("MERIDIAN NATIONAL BANK", "large")] * 18
    + [("ATLAS FINANCIAL GROUP", "large")] * 15
    + [("CROWN SAVINGS & TRUST", "large")] * 12
    + [("PACIFIC UNION BANCORP", "large")] * 10
    + [("SUMMIT CARD SERVICES", "mid")] * 7
    + [("HARBOR CREDIT UNION", "mid")] * 5
    + [("BLUE RIDGE COLLECTIONS", "mid")] * 5
    + [("NORTHSTAR RECOVERY LLC", "small")] * 3
    + [("CEDAR POINT LENDING", "small")] * 2
    + [("VERITY DATA SOLUTIONS", "small")] * 2
)

def _weighted(rng: random.Random, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def build_complaints(rng: random.Random) -> tuple[list[dict], dict]:
    issues = [i[0] for i in ISSUES]
    weights = [i[3] for i in ISSUES]
    by_issue = {i[0]: i for i in ISSUES}
    span = (END - START).days

    rows = []
    for idx in range(N_COMPLAINTS):
        # Submission dates are uniform over the window; the temporal split is
        # what makes the ordering matter downstream.
        received = START + timedelta(days=rng.randrange(span))
        issue = _weighted(rng, issues, weights)
        _issue, product, regulation, _w, escalation_prior = by_issue[issue]
        company, size = rng.choice(COMPANIES)

        amount = round(rng.lognormvariate(4.6, 1.05), 2)
        narrative = compose(
            issue, rng, amount,
            (received - timedelta(days=rng.randint(5, 90))).isoformat(),
        )

        # Company response behaviour drifts partway through the window: relief
        # rates fall after 2022-09. A model trained on a random split cannot see
        # this; a model trained on a temporal split has to live with it.
        drifted = received >= date(2022, 9, 1)
        relief_boost = 0.0 if drifted else 0.14
        size_boost = {"large": 0.10, "mid": 0.0, "small": -0.07}[size]
        # Money moves when money is at stake. Reg E and Reg Z disputes carry a
        # disputed amount; credit-reporting and collection complaints are
        # resolved with corrections and explanations instead.
        regulation_boost = {"REG_E": 0.22, "REG_Z": 0.16, "FCRA": -0.06, "FDCPA": -0.04}[regulation]
        p_monetary = min(0.62, max(0.02, 0.14 + relief_boost + size_boost + regulation_boost))
        p_nonmonetary = 0.34 if regulation in {"FCRA", "FDCPA"} else 0.12
        p_untimely = 0.02 if size == "large" else (0.08 if size == "mid" else 0.17)
        p_norelief = 0.20 if regulation == "FDCPA" else 0.10

        roll = rng.random()
        if roll < p_untimely:
            response = "Untimely response"
        elif roll < p_untimely + p_monetary:
            response = "Closed with monetary relief"
        elif roll < p_untimely + p_monetary + p_nonmonetary:
            response = "Closed with non-monetary relief"
        elif roll < p_untimely + p_monetary + p_nonmonetary + p_norelief:
            response = "Closed without relief"
        else:
            response = "Closed with explanation"

        disputed_p = (
            escalation_prior
            + (0.26 if response in {"Closed without relief", "Untimely response"} else 0.0)
            - (0.20 if response == "Closed with monetary relief" else 0.0)
            + (0.08 if amount > 800 else 0.0)
        )
        consumer_disputed = rng.random() < min(0.90, max(0.02, disputed_p))

        rows.append({
            "complaint_id": f"C{3_100_000 + idx}",
            "date_received": received.isoformat(),
            "product": product,
            "issue": issue,
            "regulation": regulation,
            "company": company,
            "company_size": size,
            "state": rng.choice(STATES),
            "submitted_via": _weighted(rng, CHANNELS, [0.68, 0.14, 0.09, 0.07, 0.02]),
            "consumer_consent": "Consent provided",
            "narrative": narrative,
            "disputed_amount": amount if regulation in {"REG_E", "REG_Z"} else "",
            "company_response": response,
            "timely_response": "No" if response == "Untimely response" else "Yes",
            "consumer_disputed": "Yes" if consumer_disputed else "No",
        })

    rows.sort(key=lambda r: r["date_received"])
    meta = {
        "response_drift_date": "2022-09-01",
        "response_drift": "relief rate falls by roughly 10 points after this date",
        "large_institution_share": round(
            sum(1 for r in rows if r["company_size"] == "large") / len(rows), 4),
    }
    return rows, meta


# --------------------------------------------------------------------------
# tabular fraud
# --------------------------------------------------------------------------

FRAUD_FEATURES = [
    "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
    "customer_age", "days_since_request", "intended_balcon_amount", "zip_count_4w",
    "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "credit_risk_score", "email_is_free",
    "phone_home_valid", "phone_mobile_valid", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "session_length_in_minutes", "device_distinct_emails_8w",
    "device_fraud_count", "keep_alive_session", "bank_months_count",
]


def build_fraud(rng: random.Random) -> tuple[list[dict], dict]:
    rows = []
    for idx in range(N_FRAUD):
        month = idx * 8 // N_FRAUD  # 0..7, ordered, for the temporal split

        income = round(min(0.99, max(0.01, rng.gauss(0.55, 0.22))), 2)
        age = rng.choice([10, 20, 20, 30, 30, 30, 40, 40, 50, 60, 70])
        credit_risk = round(rng.gauss(130 + 40 * income, 60))
        email_free = 1 if rng.random() < 0.55 else 0
        velocity_6h = round(max(0.0, rng.gauss(5000, 3000)), 1)
        velocity_24h = round(max(0.0, velocity_6h * rng.uniform(3.0, 5.5)), 1)
        device_emails = rng.choices([1, 1, 1, 2, 3, 5, 9], weights=[50, 20, 10, 8, 6, 4, 2])[0]
        session = round(max(0.1, rng.gauss(6.0, 3.4)), 2)
        foreign = 1 if rng.random() < 0.03 else 0
        keep_alive = 1 if rng.random() < 0.58 else 0
        dob_emails = rng.randint(0, 40)

        # The signal. keep_alive_session flips sign at month 4: early in the
        # window keeping the session alive is a legitimate-user habit, later it
        # is automation. Any model validated on a random split will not notice.
        logit = (
            -7.72
            + 2.30 * (device_emails >= 3)
            + 1.60 * (velocity_24h > 22000)
            + 1.45 * foreign
            + 0.95 * email_free
            + 1.20 * (age <= 20)
            - 1.70 * (credit_risk > 160)
            - 0.85 * (session > 8)
            + 1.10 * (dob_emails >= 12)
            + ((-0.65 if month < 4 else 0.80) * keep_alive)
        )
        fraud = 1 if rng.random() < 1 / (1 + math.exp(-logit)) else 0

        rows.append({
            "application_id": f"A{700_000 + idx}",
            "month": month,
            "income": income,
            "name_email_similarity": round(rng.random(), 3),
            "prev_address_months_count": rng.choice([-1, 6, 12, 24, 48, 96]),
            "current_address_months_count": rng.choice([-1, 3, 9, 18, 36, 72, 120]),
            "customer_age": age,
            "days_since_request": round(rng.expovariate(1 / 2.5), 3),
            "intended_balcon_amount": round(max(-1.0, rng.gauss(8.0, 12.0)), 2),
            "zip_count_4w": rng.randint(50, 5500),
            "velocity_6h": velocity_6h,
            "velocity_24h": velocity_24h,
            "velocity_4w": round(max(0.0, velocity_24h * rng.uniform(0.7, 1.4)), 1),
            "bank_branch_count_8w": rng.randint(0, 2400),
            "date_of_birth_distinct_emails_4w": dob_emails,
            "credit_risk_score": credit_risk,
            "email_is_free": email_free,
            "phone_home_valid": 1 if rng.random() < 0.42 else 0,
            "phone_mobile_valid": 1 if rng.random() < 0.89 else 0,
            "has_other_cards": 1 if rng.random() < 0.22 else 0,
            "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000]),
            "foreign_request": foreign,
            "session_length_in_minutes": session,
            "device_distinct_emails_8w": device_emails,
            "device_fraud_count": rng.choices([0, 1, 2], weights=[95, 4, 1])[0],
            "keep_alive_session": keep_alive,
            "bank_months_count": rng.choice([-1, 3, 8, 14, 22, 31]),
            "fraud_bool": fraud,
        })

    rate = sum(r["fraud_bool"] for r in rows) / len(rows)
    meta = {
        "base_rate": round(rate, 5),
        "concept_drift_month": 4,
        "drifting_feature": "keep_alive_session",
        "drift_description": "the sign of the keep_alive_session coefficient inverts at month 4",
    }
    return rows, meta


def main() -> None:
    rng = random.Random(SEED)
    RAW.mkdir(parents=True, exist_ok=True)

    complaints, complaint_meta = build_complaints(rng)
    with (RAW / "complaints.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(complaints[0]), **CSV_DIALECT)
        writer.writeheader()
        writer.writerows(complaints)

    fraud, fraud_meta = build_fraud(rng)
    with (RAW / "fraud_applications.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fraud[0]), **CSV_DIALECT)
        writer.writeheader()
        writer.writerows(fraud)

    (RAW / "meta.json").write_text(
        json.dumps({"seed": SEED, "complaints": complaint_meta, "fraud": fraud_meta}, indent=2),
        encoding="utf-8", newline="\n",
    )

    print(f"complaints        {len(complaints):>6} rows, {len({r['issue'] for r in complaints})} issue classes")
    print(f"                  {complaint_meta['large_institution_share']:.1%} from large institutions")
    print(f"fraud             {len(fraud):>6} rows, base rate {fraud_meta['base_rate']:.3%}")
    print(f"                  concept drift on {fraud_meta['drifting_feature']} at month {fraud_meta['concept_drift_month']}")


if __name__ == "__main__":
    main()
