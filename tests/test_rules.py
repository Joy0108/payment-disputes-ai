"""The rules engine. Everything here is a date somebody could be sued over."""

from __future__ import annotations

from datetime import date

import pytest

from disputes.eval.deadlines import CASES, run
from disputes.rules.calendar import (
    add_business_days,
    business_days_between,
    federal_holidays,
    is_business_day,
    next_business_day,
)
from disputes.rules.deadlines import fcra, fdcpa, regulation_e, regulation_e_liability, regulation_for_issue, regulation_z
from disputes.rules.validate import map_reason_code, validate_amounts, validate_dates

# --- the hand-computed oracle ----------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"][:60])
def test_hand_computed_deadline_case(case):
    """Each expectation was worked out from the regulation and a calendar.

    A test whose expected value came from the implementation proves only that
    the implementation agrees with itself.
    """
    assert case["compute"]() == case["expected"], case["why"]


def test_deadline_exactness_is_absolute():
    report = run()
    assert report["exactness"] == 1.0, report["failures"]


# --- calendar ---------------------------------------------------------------

def test_weekends_are_not_business_days():
    assert not is_business_day(date(2024, 11, 23))  # Saturday
    assert not is_business_day(date(2024, 11, 24))  # Sunday
    assert is_business_day(date(2024, 11, 25))


def test_thanksgiving_is_the_fourth_thursday():
    assert date(2024, 11, 28) in federal_holidays(2024)
    assert date(2023, 11, 23) in federal_holidays(2023)


def test_a_fixed_holiday_on_a_sunday_is_observed_on_the_monday():
    # 4 July 2021 fell on a Sunday.
    assert date(2021, 7, 5) in federal_holidays(2021)
    assert date(2021, 7, 4) not in federal_holidays(2021)


def test_new_year_falling_on_a_saturday_is_observed_on_31_december():
    # 1 January 2022 was a Saturday, observed Friday 31 December 2021.
    assert date(2021, 12, 31) in federal_holidays(2021)


def test_counting_starts_the_day_after_the_notice():
    """Day zero is the day of receipt. This off-by-one is a real violation."""
    assert add_business_days(date(2024, 11, 20), 1) == date(2024, 11, 21)
    assert add_business_days(date(2024, 11, 20), 0) == date(2024, 11, 20)


def test_business_days_between_is_antisymmetric():
    a, b = date(2024, 11, 1), date(2024, 12, 1)
    assert business_days_between(a, b) == -business_days_between(b, a)


def test_next_business_day_skips_a_holiday_weekend():
    # Friday 29 November 2024 is a business day; the Thursday was Thanksgiving.
    assert next_business_day(date(2024, 11, 27)) == date(2024, 11, 29)


# --- Regulation E -----------------------------------------------------------

def test_without_provisional_credit_there_is_no_extension():
    result = regulation_e("2024-11-20", "2024-11-01")
    names = [d.name for d in result.deadlines]
    assert "extended investigation deadline" not in names
    assert any("No provisional credit" in f for f in result.findings)


def test_provisional_credit_unlocks_the_extension_and_its_own_notice_deadline():
    result = regulation_e("2024-11-20", "2024-11-01", provisional_credit_given=True)
    names = [d.name for d in result.deadlines]
    assert "extended investigation deadline" in names
    assert "notify consumer of the provisional credit" in names


def test_new_account_and_point_of_sale_do_not_extend_the_same_period():
    """The 20-business-day initial period is a new-account rule only."""
    pos = regulation_e("2024-11-20", "2024-11-01", point_of_sale=True, provisional_credit_given=True)
    new = regulation_e("2024-11-20", "2024-11-01", account_opened="2024-11-05", provisional_credit_given=True)
    initial = "investigation concluded, or provisional credit given"
    assert _due(pos, initial) != _due(new, initial)
    # Both reach 90 days on the outer period, by different routes.
    assert _due(pos, "extended investigation deadline") == _due(new, "extended investigation deadline")


def test_late_notice_stops_the_calculation_and_says_why():
    result = regulation_e("2025-01-05", "2024-11-01")
    assert result.timely is False
    assert len(result.deadlines) == 1
    assert "after the 60-day window" in result.findings[0]


def test_liability_tiers_use_two_different_clocks():
    """Business days from discovery for tiers 1 and 2; calendar days from the
    statement for tier 3. Conflating them is the usual error."""
    tier1 = regulation_e_liability("2024-11-18", "2024-11-19", "2024-11-01")
    tier2 = regulation_e_liability("2024-11-01", "2024-11-20", "2024-11-01")
    tier3 = regulation_e_liability("2024-08-01", "2025-01-15", "2024-11-01")
    assert (tier1["liability_cap_usd"], tier2["liability_cap_usd"], tier3["liability_cap_usd"]) == (50.0, 500.0, None)


def test_a_non_access_device_transfer_has_no_fifty_dollar_tier():
    result = regulation_e_liability("2024-11-18", "2024-11-19", "2024-11-01", access_device=False)
    assert result["liability_cap_usd"] == 0.0
    assert result["citation"] == "1005.6(b)(3)"


# --- Regulation Z -----------------------------------------------------------

def test_the_ninety_day_cap_binds_over_long_billing_cycles():
    short = regulation_z("2024-11-20", "2024-11-01", billing_cycle_days=30)
    long = regulation_z("2024-11-20", "2024-11-01", billing_cycle_days=50)
    assert _due(short, "resolution complete") < _due(long, "resolution complete")
    assert "90-day cap binds" in _note(long, "resolution complete")


def test_reg_z_records_the_collection_prohibition():
    result = regulation_z("2024-11-20", "2024-11-01")
    assert any("may not attempt to collect" in f for f in result.findings)
    assert "1026.13(d)" in result.citations


# --- FCRA and FDCPA ---------------------------------------------------------

def test_fcra_extension_requires_additional_information():
    assert _due(fcra("2024-11-20"), "reinvestigation complete") == "2024-12-20"
    assert _due(fcra("2024-11-20", additional_information_provided=True), "reinvestigation complete") == "2025-01-04"


def test_fdcpa_dispute_inside_the_window_triggers_cessation():
    inside = fdcpa("2024-11-20", written_dispute_date="2024-12-01")
    assert inside.timely
    assert any("cease" in f for f in inside.findings)
    assert "1692g(b)" in inside.citations


def test_fdcpa_dispute_outside_the_window_does_not():
    outside = fdcpa("2024-11-20", written_dispute_date="2025-01-15")
    assert not outside.timely
    assert "1692e" in outside.citations


def test_every_issue_maps_to_a_regulation():
    from disputes.config import COMPLAINT_ISSUES

    for issue in COMPLAINT_ISSUES:
        assert regulation_for_issue(issue) in {"REG_E", "REG_Z", "FCRA", "FDCPA"}


# --- validation -------------------------------------------------------------

def test_amounts_must_reconcile_to_the_transactions():
    ok = validate_amounts(100.0, [{"id": "a", "amount": 60.0}, {"id": "b", "amount": 40.0}])
    assert ok.ok
    bad = validate_amounts(100.0, [{"id": "a", "amount": 60.0}])
    assert not bad.ok
    assert "does not reconcile" in bad.issues[0].message


def test_a_statement_cannot_precede_its_transaction():
    result = validate_dates("2024-11-10", "2024-11-01", "2024-11-20")
    assert not result.ok
    assert any("before the transaction" in i.message for i in result.issues)


def test_notice_cannot_precede_the_statement():
    result = validate_dates("2024-10-01", "2024-11-01", "2024-10-15")
    assert not result.ok
    assert any("precedes the statement" in i.message for i in result.issues)


def test_a_future_date_is_an_error():
    result = validate_dates("2099-01-01", "2099-01-02", "2099-01-03")
    assert not result.ok


def test_an_unknown_reason_code_is_surfaced_not_guessed():
    result = map_reason_code("99.9")
    assert result["mapped"] is False
    assert "route to a human" in result["reason"]


def test_a_reason_code_can_be_derived_from_the_issue():
    result = map_reason_code(None, "Unauthorized transactions or other transaction problem")
    assert result["mapped"] and result["derived"]
    assert result["regulation"] == "REG_E"


def test_an_issue_with_no_card_network_route_is_not_forced_into_one():
    result = map_reason_code(None, "Incorrect information on your report")
    assert result["mapped"] is False


def _due(result, name: str) -> str:
    return next(d.due.isoformat() for d in result.deadlines if d.name == name)


def _note(result, name: str) -> str:
    return next(d.note for d in result.deadlines if d.name == name)
