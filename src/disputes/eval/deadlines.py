"""Hand-computed deadline cases.

Every expected value here was worked out by hand from the regulation text and a
calendar, not produced by the code and pasted back. A test whose expectation
came from the implementation proves the implementation is self-consistent and
nothing else.

The cases are chosen to cover the interactions rather than the happy path: a
notice period spanning Thanksgiving, a new account that is *also* a
point-of-sale transfer, the 90-day cap binding before two billing cycles
elapse, and the boundary where a consumer notice is exactly on time.

This is gated absolutely. A deadline is right or it is a regulatory violation,
and there is no threshold at which some of them being wrong is acceptable.
"""

from __future__ import annotations

from typing import Any

from ..rules import deadlines as rules
from ..rules.calendar import add_business_days, federal_holidays

# (label, callable, expected)
CASES: list[dict[str, Any]] = [
    # --- business-day arithmetic ------------------------------------------
    {
        "case": "10 business days from Wed 20 Nov 2024 crosses Thanksgiving",
        "kind": "calendar",
        "compute": lambda: add_business_days(_d("2024-11-20"), 10).isoformat(),
        "expected": "2024-12-05",
        "why": "21,22,25,26,27 then 28 Nov is Thanksgiving, 29, then 2-5 Dec. Ten counted days land on 5 December.",
    },
    {
        "case": "3 business days from Fri 27 Dec 2024, with 1 Jan 2025 a holiday",
        "kind": "calendar",
        "compute": lambda: add_business_days(_d("2024-12-27"), 3).isoformat(),
        "expected": "2025-01-02",
        "why": "30 and 31 December count, 1 January is a federal holiday, so the third day is 2 January.",
    },
    {
        "case": "Juneteenth is not a federal holiday in 2020",
        "kind": "calendar",
        "compute": lambda: _d("2020-06-19") in federal_holidays(2020),
        "expected": False,
        "why": "Juneteenth became a federal holiday in June 2021. A 2020 date must not skip 19 June.",
    },
    {
        "case": "Juneteenth 2021 falls on a Saturday and is observed on Friday 18 June",
        "kind": "calendar",
        "compute": lambda: _d("2021-06-18") in federal_holidays(2021),
        "expected": True,
        "why": "A fixed-date holiday on a Saturday is observed the preceding Friday.",
    },

    # --- Regulation E ------------------------------------------------------
    {
        "case": "Reg E: 10 business days when no provisional credit is given",
        "kind": "reg_e",
        "compute": lambda: _due(rules.regulation_e("2024-11-20", "2024-11-01"),
                                "investigation concluded, or provisional credit given"),
        "expected": "2024-12-05",
        "why": "1005.11(c)(1). Without provisional credit the extension is unavailable.",
    },
    {
        "case": "Reg E: extension to 45 calendar days once provisional credit is given",
        "kind": "reg_e",
        "compute": lambda: _due(rules.regulation_e("2024-11-20", "2024-11-01", provisional_credit_given=True),
                                "extended investigation deadline"),
        "expected": "2025-01-04",
        "why": "1005.11(c)(2). 45 calendar days from 20 November 2024.",
    },
    {
        "case": "Reg E: point of sale gets 90 days, not 45",
        "kind": "reg_e",
        "compute": lambda: _due(rules.regulation_e("2024-11-20", "2024-11-01", point_of_sale=True,
                                                   provisional_credit_given=True),
                                "extended investigation deadline"),
        "expected": "2025-02-18",
        "why": "1005.11(c)(3). 90 calendar days from 20 November 2024.",
    },
    {
        "case": "Reg E: a point-of-sale transfer does NOT extend the initial period to 20 business days",
        "kind": "reg_e",
        "compute": lambda: _due(rules.regulation_e("2024-11-20", "2024-11-01", point_of_sale=True,
                                                   provisional_credit_given=True),
                                "investigation concluded, or provisional credit given"),
        "expected": "2024-12-05",
        "why": "The 20-business-day initial period is a new-account rule only. POS extends the outer period alone.",
    },
    {
        "case": "Reg E: a new account gets 20 business days",
        "kind": "reg_e",
        "compute": lambda: _due(rules.regulation_e("2024-11-20", "2024-11-01", account_opened="2024-11-05"),
                                "investigation concluded, or provisional credit given"),
        "expected": "2024-12-19",
        "why": "1005.11(c)(3). 20 business days from 20 November 2024, skipping Thanksgiving and the weekends.",
    },
    {
        "case": "Reg E: consumer notice exactly 60 calendar days after the statement is timely",
        "kind": "reg_e",
        "compute": lambda: rules.regulation_e("2024-12-31", "2024-11-01").timely,
        "expected": True,
        "why": "1005.11(b) gives 60 days from transmittal. 1 November plus 60 days is 31 December, inclusive.",
    },
    {
        "case": "Reg E: notice on day 61 is out of time",
        "kind": "reg_e",
        "compute": lambda: rules.regulation_e("2025-01-01", "2024-11-01").timely,
        "expected": False,
        "why": "One day past the window; the 1005.11 obligations are not triggered.",
    },

    # --- Regulation E liability -------------------------------------------
    {
        "case": "Reg E liability: notice within two business days of discovery caps at $50",
        "kind": "reg_e_liability",
        "compute": lambda: rules.regulation_e_liability("2024-11-18", "2024-11-19", "2024-11-01")["liability_cap_usd"],
        "expected": 50.0,
        "why": "1005.6(b) tier 1.",
    },
    {
        "case": "Reg E liability: later than two business days but inside 60 days caps at $500",
        "kind": "reg_e_liability",
        "compute": lambda: rules.regulation_e_liability("2024-11-01", "2024-11-20", "2024-11-01")["liability_cap_usd"],
        "expected": 500.0,
        "why": "1005.6(b) tier 2.",
    },
    {
        "case": "Reg E liability: past the 60-day statement window is uncapped",
        "kind": "reg_e_liability",
        "compute": lambda: rules.regulation_e_liability("2024-08-01", "2025-01-15", "2024-11-01")["liability_cap_usd"],
        "expected": None,
        "why": "1005.6(b) tier 3: unlimited for transfers after the 60 days and before notice.",
    },

    # --- Regulation Z ------------------------------------------------------
    {
        "case": "Reg Z: two 30-day billing cycles resolve inside the 90-day cap",
        "kind": "reg_z",
        "compute": lambda: _due(rules.regulation_z("2024-11-20", "2024-11-01"), "resolution complete"),
        "expected": "2025-01-19",
        "why": "1026.13(c)(2). Two 30-day cycles is 60 days, which is inside the 90-day cap.",
    },
    {
        "case": "Reg Z: the 90-day cap binds when billing cycles are 50 days",
        "kind": "reg_z",
        "compute": lambda: _due(rules.regulation_z("2024-11-20", "2024-11-01", billing_cycle_days=50),
                                "resolution complete"),
        "expected": "2025-02-18",
        "why": "Two cycles would be 100 days; 1026.13(c)(2) caps resolution at 90.",
    },
    {
        "case": "Reg Z: acknowledgement is due 30 calendar days after the notice",
        "kind": "reg_z",
        "compute": lambda: _due(rules.regulation_z("2024-11-20", "2024-11-01"),
                                "written acknowledgement to the consumer"),
        "expected": "2024-12-20",
        "why": "1026.13(c)(1).",
    },

    # --- FCRA and FDCPA ----------------------------------------------------
    {
        "case": "FCRA: reinvestigation is due 30 calendar days after the dispute",
        "kind": "fcra",
        "compute": lambda: _due(rules.fcra("2024-11-20"), "reinvestigation complete"),
        "expected": "2024-12-20",
        "why": "1681i(a)(1).",
    },
    {
        "case": "FCRA: extended to 45 days when the consumer supplies more information",
        "kind": "fcra",
        "compute": lambda: _due(rules.fcra("2024-11-20", additional_information_provided=True),
                                "reinvestigation complete"),
        "expected": "2025-01-04",
        "why": "1681i(a)(1) permits the extension to 45 days.",
    },
    {
        "case": "FDCPA: validation notice is due 5 calendar days after first contact",
        "kind": "fdcpa",
        "compute": lambda: _due(rules.fdcpa("2024-11-20"), "validation notice sent"),
        "expected": "2024-11-25",
        "why": "1692g(a).",
    },
    {
        "case": "FDCPA: a written dispute on day 30 is inside the window",
        "kind": "fdcpa",
        "compute": lambda: rules.fdcpa("2024-11-20", written_dispute_date="2024-12-20").timely,
        "expected": True,
        "why": "1692g(b). Thirty days from 20 November 2024 is 20 December.",
    },
    {
        "case": "FDCPA: a written dispute on day 31 is outside it",
        "kind": "fdcpa",
        "compute": lambda: rules.fdcpa("2024-11-20", written_dispute_date="2024-12-21").timely,
        "expected": False,
        "why": "One day past the validation window; the cessation requirement is not triggered.",
    },
]


def _d(value: str):
    from datetime import date

    return date.fromisoformat(value)


def _due(result, name: str) -> str:
    for deadline in result.deadlines:
        if deadline.name == name:
            return deadline.due.isoformat()
    raise KeyError(f"{name!r} not among {[d.name for d in result.deadlines]}")


def run() -> dict[str, Any]:
    rows = []
    for case in CASES:
        try:
            actual = case["compute"]()
            passed = actual == case["expected"]
            error = None
        except Exception as exc:
            actual, passed, error = None, False, f"{type(exc).__name__}: {exc}"
        rows.append({
            "case": case["case"], "kind": case["kind"], "expected": case["expected"],
            "actual": actual, "passed": passed, "why": case["why"], "error": error,
        })

    passed = sum(1 for r in rows if r["passed"])
    return {
        "cases": len(rows),
        "passed": passed,
        "exactness": round(passed / len(rows), 4) if rows else float("nan"),
        "failures": [r for r in rows if not r["passed"]],
        "by_kind": {
            kind: sum(1 for r in rows if r["kind"] == kind and r["passed"])
            for kind in sorted({r["kind"] for r in rows})
        },
        "rows": rows,
    }
