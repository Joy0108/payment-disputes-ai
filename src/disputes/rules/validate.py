"""Field validation and reason-code mapping.

Two jobs that look clerical and are not:

**Amount and date validation.** A dispute whose amount does not reconcile to
the transactions being disputed, or whose dates are impossible, produces a
correct-looking response about the wrong facts. These are the checks that stop
a generated letter quoting a figure nobody can tie to a statement line.

**Reason-code mapping.** The card network reason code determines the evidence
the issuer must supply and the window it has to supply it in. The mapping is a
table with citations, not an inference, and an unmapped code is surfaced rather
than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .calendar import parse_date

# Visa and Mastercard dispute reason codes. Only the ones a consumer dispute
# workflow actually routes; anything else is returned as unmapped so it reaches
# a human rather than being silently mapped to the nearest neighbour.
REASON_CODES = {
    "10.4": {
        "network": "Visa", "label": "Other Fraud - Card Absent Environment",
        "category": "fraud", "issuer_window_days": 120, "regulation": "REG_E",
        "evidence": ["cardholder statement of unauthorised use", "transaction detail", "device and IP data"],
    },
    "10.1": {
        "network": "Visa", "label": "EMV Liability Shift Counterfeit Fraud",
        "category": "fraud", "issuer_window_days": 120, "regulation": "REG_E",
        "evidence": ["terminal capability data", "cardholder statement"],
    },
    "13.1": {
        "network": "Visa", "label": "Merchandise/Services Not Received",
        "category": "consumer_dispute", "issuer_window_days": 120, "regulation": "REG_Z",
        "evidence": ["expected delivery date", "proof of non-delivery", "attempt to resolve with the merchant"],
    },
    "13.3": {
        "network": "Visa", "label": "Not as Described or Defective Merchandise",
        "category": "consumer_dispute", "issuer_window_days": 120, "regulation": "REG_Z",
        "evidence": ["description as advertised", "condition on receipt", "return or attempted return"],
    },
    "13.6": {
        "network": "Visa", "label": "Credit Not Processed",
        "category": "consumer_dispute", "issuer_window_days": 120, "regulation": "REG_Z",
        "evidence": ["merchant refund policy", "proof of return", "date credit was promised"],
    },
    "13.7": {
        "network": "Visa", "label": "Cancelled Merchandise/Services",
        "category": "consumer_dispute", "issuer_window_days": 120, "regulation": "REG_Z",
        "evidence": ["cancellation date and method", "merchant cancellation terms"],
    },
    "12.5": {
        "network": "Visa", "label": "Incorrect Amount",
        "category": "processing_error", "issuer_window_days": 120, "regulation": "REG_Z",
        "evidence": ["receipt showing the agreed amount", "statement entry"],
    },
    "12.6.1": {
        "network": "Visa", "label": "Duplicate Processing",
        "category": "processing_error", "issuer_window_days": 120, "regulation": "REG_Z",
        "evidence": ["both statement entries", "single receipt"],
    },
    "4837": {
        "network": "Mastercard", "label": "No Cardholder Authorisation",
        "category": "fraud", "issuer_window_days": 120, "regulation": "REG_E",
        "evidence": ["cardholder statement of unauthorised use", "transaction detail"],
    },
    "4853": {
        "network": "Mastercard", "label": "Cardholder Dispute",
        "category": "consumer_dispute", "issuer_window_days": 120, "regulation": "REG_Z",
        "evidence": ["merchant contact attempt", "supporting documentation of the dispute"],
    },
    "4834": {
        "network": "Mastercard", "label": "Point-of-Interaction Error",
        "category": "processing_error", "issuer_window_days": 90, "regulation": "REG_Z",
        "evidence": ["receipt", "statement entry", "terminal data"],
    },
}

ISSUE_TO_REASON_CODE = {
    "Unauthorized transactions or other transaction problem": "10.4",
    "Fraud or scam": "10.4",
    "Problem with a purchase shown on your statement": "13.1",
    "Charged fees or interest you didn't expect": "12.5",
    "Trouble using your card": "4834",
    "Problem with a lender or other company charging your account": "12.6.1",
}

_MONEY = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)")


@dataclass
class ValidationIssue:
    field: str
    severity: str  # error | warning
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "severity": self.severity, "message": self.message}


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)
    normalised: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": [i.to_dict() for i in self.issues], "normalised": self.normalised}


def validate_amounts(
    disputed_amount: float | str | None,
    transactions: list[dict[str, Any]] | None = None,
    tolerance: float = 0.01,
) -> ValidationResult:
    result = ValidationResult()

    amount = _to_amount(disputed_amount)
    if amount is None:
        result.issues.append(ValidationIssue("disputed_amount", "error", "the disputed amount is missing or unparseable"))
        return result
    if amount <= 0:
        result.issues.append(ValidationIssue("disputed_amount", "error", f"the disputed amount must be positive, got {amount}"))
        return result
    result.normalised["disputed_amount"] = round(amount, 2)

    if transactions:
        total = round(sum(_to_amount(t.get("amount")) or 0.0 for t in transactions), 2)
        result.normalised["transaction_total"] = total
        if abs(total - amount) > tolerance:
            result.issues.append(ValidationIssue(
                "disputed_amount", "error",
                f"the disputed amount {amount:.2f} does not reconcile to the {len(transactions)} transaction(s) "
                f"listed, which total {total:.2f}"))
        for txn in transactions:
            if _to_amount(txn.get("amount")) is None:
                result.issues.append(ValidationIssue(
                    "transactions", "error", f"transaction {txn.get('id', '?')} has no parseable amount"))
    return result


def validate_dates(
    transaction_date: str | date | None,
    statement_date: str | date | None,
    notice_date: str | date | None,
    *,
    today: date | None = None,
) -> ValidationResult:
    """Ordering and plausibility. Each check exists because the ordering it
    enforces is a precondition for a deadline calculation downstream."""
    result = ValidationResult()
    today = today or date.today()

    parsed: dict[str, date | None] = {}
    for name, value in (("transaction_date", transaction_date), ("statement_date", statement_date),
                        ("notice_date", notice_date)):
        if value in (None, ""):
            parsed[name] = None
            result.issues.append(ValidationIssue(name, "error", "date is missing"))
            continue
        try:
            parsed[name] = parse_date(value)
        except ValueError:
            parsed[name] = None
            result.issues.append(ValidationIssue(name, "error", f"{value!r} is not an ISO date"))

    for name, day in parsed.items():
        if day is None:
            continue
        result.normalised[name] = day.isoformat()
        if day > today:
            result.issues.append(ValidationIssue(name, "error", f"{day.isoformat()} is in the future"))
        if day.year < 2000:
            result.issues.append(ValidationIssue(name, "warning", f"{day.isoformat()} is implausibly old"))

    txn, stmt, notice = parsed["transaction_date"], parsed["statement_date"], parsed["notice_date"]
    if txn and stmt and txn > stmt:
        result.issues.append(ValidationIssue(
            "statement_date", "error",
            f"the statement was transmitted on {stmt.isoformat()}, before the transaction on {txn.isoformat()}; "
            "the 60-day clock cannot start before the transaction it reflects"))
    if stmt and notice and notice < stmt:
        result.issues.append(ValidationIssue(
            "notice_date", "error",
            f"notice on {notice.isoformat()} precedes the statement on {stmt.isoformat()}"))
    if txn and notice and (notice - txn).days > 540:
        result.issues.append(ValidationIssue(
            "notice_date", "warning",
            f"{(notice - txn).days} days elapsed between the transaction and the notice, which is outside every "
            "network chargeback window even where the regulatory claim survives"))
    return result


def map_reason_code(code: str | None, issue: str | None = None) -> dict[str, Any]:
    """Resolve a network reason code, or derive one from the issue.

    An unmapped code returns ``mapped: False`` rather than a best guess. The
    reason code drives the evidence the issuer must gather; supplying the wrong
    evidence loses the dispute on procedure rather than on the merits.
    """
    if code:
        entry = REASON_CODES.get(str(code).strip())
        if entry is None:
            return {
                "mapped": False, "code": code,
                "reason": "reason code not in the mapping table; route to a human rather than assume the nearest match",
            }
        return {"mapped": True, "code": code, "derived": False, **entry}

    derived = ISSUE_TO_REASON_CODE.get(issue or "")
    if derived is None:
        return {
            "mapped": False, "code": None, "issue": issue,
            "reason": "no reason code supplied and the issue does not map to one; a card-network dispute may not apply",
        }
    return {"mapped": True, "code": derived, "derived": True, **REASON_CODES[derived]}


def _to_amount(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _MONEY.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None
