"""Regulatory deadline computation.

This module is deliberately not a model, and not a prompt. Every deadline here
is a mechanical consequence of a date and a fact pattern, and the failure mode
of getting one wrong is a regulatory violation rather than a lower metric. A
language model asked to compute a Regulation E provisional-credit deadline will
usually be right, and "usually" is the wrong standard for a date that determines
whether an institution owes a consumer money.

So the LLM's job upstream is to *extract the facts* - when was notice given,
was the account new, was it a point-of-sale transaction - and the arithmetic
happens here, with the citation for every branch attached to its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .calendar import add_business_days, add_calendar_days, business_days_between, parse_date

# The regulation each issue is resolved under. A mis-mapped issue produces a
# deadline computed under the wrong statute, which is why this table lives with
# the arithmetic rather than in the classifier.
REGULATION_BY_ISSUE = {
    "Unauthorized transactions or other transaction problem": "REG_E",
    "Fraud or scam": "REG_E",
    "Managing an account": "REG_E",
    "Closing an account": "REG_E",
    "Problem caused by your funds being low": "REG_E",
    "Trouble using your card": "REG_E",
    "Problem with a lender or other company charging your account": "REG_E",
    "Money was not available when promised": "REG_E",
    "Confusing or missing disclosures": "REG_E",
    "Problem with a purchase shown on your statement": "REG_Z",
    "Other features, terms, or problems": "REG_Z",
    "Getting a credit card": "REG_Z",
    "Charged fees or interest you didn't expect": "REG_Z",
    "Struggling to repay your loan": "REG_Z",
    "Incorrect information on your report": "FCRA",
    "Problem with a company's investigation into an existing problem": "FCRA",
    "Improper use of your report": "FCRA",
    "Attempts to collect debt not owed": "FDCPA",
    "Written notification about debt": "FDCPA",
    "Communication tactics": "FDCPA",
}


@dataclass
class Deadline:
    name: str
    due: date
    basis: str          # "business days" | "calendar days" | "billing cycles"
    count: int
    citation: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "due": self.due.isoformat(),
            "basis": f"{self.count} {self.basis}",
            "citation": self.citation,
            "note": self.note,
        }


@dataclass
class DeadlineResult:
    regulation: str
    timely: bool
    deadlines: list[Deadline] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regulation": self.regulation,
            "consumer_notice_timely": self.timely,
            "deadlines": [d.to_dict() for d in self.deadlines],
            "findings": self.findings,
            "citations": sorted(set(self.citations)),
        }


# ---------------------------------------------------------------------------
# Regulation E
# ---------------------------------------------------------------------------

def regulation_e(
    notice_date: str | date,
    statement_date: str | date,
    *,
    account_opened: str | date | None = None,
    point_of_sale: bool = False,
    foreign_initiated: bool = False,
    provisional_credit_given: bool = False,
) -> DeadlineResult:
    """Error resolution timing under 12 CFR 1005.11.

    Three branches interact and they are easy to get wrong in combination:

    * a *new account* - one where the transfer occurred within 30 days of the
      first deposit - gets 20 business days instead of 10, and 90 days instead
      of 45;
    * a point-of-sale or foreign-initiated transfer gets the 90-day extension
      but **not** the 20-business-day initial period;
    * the extension is only available *if* provisional credit is given within
      the initial 10 business days. Without it the institution is held to the
      shorter period regardless of how complex the investigation is.
    """
    notice = parse_date(notice_date)
    statement = parse_date(statement_date)
    result = DeadlineResult(regulation="REG_E", timely=True)

    # 1005.11(b): notice must arrive within 60 calendar days of transmittal.
    notice_deadline = add_calendar_days(statement, 60)
    result.timely = notice <= notice_deadline
    result.citations.append("1005.11(b)")
    if not result.timely:
        days_late = (notice - notice_deadline).days
        result.findings.append(
            f"Consumer notice arrived {days_late} day(s) after the 60-day window that closed on "
            f"{notice_deadline.isoformat()}; the error resolution obligations under 1005.11 are not triggered, "
            "though the institution may still investigate voluntarily."
        )
        result.deadlines.append(Deadline(
            "consumer notice deadline (missed)", notice_deadline, "calendar days", 60, "1005.11(b)"))
        return result

    result.deadlines.append(Deadline(
        "consumer notice deadline", notice_deadline, "calendar days", 60, "1005.11(b)",
        f"notice received {(notice_deadline - notice).days} day(s) inside the window"))

    is_new_account = False
    if account_opened is not None:
        opened = parse_date(account_opened)
        is_new_account = 0 <= (notice - opened).days <= 30
        if is_new_account:
            result.findings.append(
                "The transfer occurred within 30 days of the first deposit, so the new-account periods apply.")

    initial_days = 20 if is_new_account else 10
    initial_due = add_business_days(notice, initial_days)
    result.deadlines.append(Deadline(
        "investigation concluded, or provisional credit given", initial_due, "business days", initial_days,
        "1005.11(c)(3)" if is_new_account else "1005.11(c)(1)"))
    result.citations.append("1005.11(c)(3)" if is_new_account else "1005.11(c)(1)")

    extended_days = 90 if (is_new_account or point_of_sale or foreign_initiated) else 45
    reason = (
        "new account" if is_new_account
        else "point-of-sale transfer" if point_of_sale
        else "transfer initiated outside the United States" if foreign_initiated
        else ""
    )
    if provisional_credit_given:
        extended_due = add_calendar_days(notice, extended_days)
        result.deadlines.append(Deadline(
            "extended investigation deadline", extended_due, "calendar days", extended_days,
            "1005.11(c)(3)" if extended_days == 90 else "1005.11(c)(2)",
            f"available because provisional credit was given{f'; {extended_days}-day period applies ({reason})' if reason else ''}"))
        result.citations.append("1005.11(c)(3)" if extended_days == 90 else "1005.11(c)(2)")

        notify_due = add_business_days(notice, 2)
        result.deadlines.append(Deadline(
            "notify consumer of the provisional credit", notify_due, "business days", 2, "1005.11(c)(2)"))
    else:
        result.findings.append(
            f"No provisional credit recorded, so the extension to {extended_days} days is not available and the "
            f"investigation must conclude within {initial_days} business days, by {initial_due.isoformat()}."
        )

    # 1005.11(d): report the result within three business days of concluding.
    conclusion = add_calendar_days(notice, extended_days) if provisional_credit_given else initial_due
    result.deadlines.append(Deadline(
        "report results to the consumer", add_business_days(conclusion, 3), "business days", 3, "1005.11(d)",
        "measured from the date the investigation concludes"))
    result.deadlines.append(Deadline(
        "correct the error if one occurred", add_business_days(conclusion, 1), "business days", 1, "1005.11(c)(1)"))
    result.citations.extend(["1005.11(d)", "1005.11(c)(1)"])
    return result


def regulation_e_liability(
    discovery_date: str | date,
    notice_date: str | date,
    statement_date: str | date,
    *,
    access_device: bool = True,
) -> dict[str, Any]:
    """The 1005.6 liability tier the facts put the consumer in.

    Three tiers, and which one applies turns on two different clocks: business
    days from *discovery* for the access-device tiers, and calendar days from
    *statement transmittal* for the unlimited-liability tier. They are not the
    same clock and conflating them is the usual error.
    """
    discovery = parse_date(discovery_date)
    notice = parse_date(notice_date)
    statement = parse_date(statement_date)

    business_days = business_days_between(discovery, notice)
    statement_days = (notice - statement).days

    if not access_device:
        if statement_days <= 60:
            tier, cap, citation = "no liability for transfers reported within 60 days", 0.0, "1005.6(b)(3)"
        else:
            tier, cap, citation = "unlimited for transfers after the 60-day window", None, "1005.6(b)(3)"
    elif business_days <= 2:
        tier, cap, citation = "tier 1", 50.0, "1005.6(b)"
    elif statement_days <= 60:
        tier, cap, citation = "tier 2", 500.0, "1005.6(b)"
    else:
        tier, cap, citation = "tier 3", None, "1005.6(b)"

    return {
        "tier": tier,
        "liability_cap_usd": cap,
        "business_days_from_discovery_to_notice": business_days,
        "calendar_days_from_statement_to_notice": statement_days,
        "citation": citation,
        "note": (
            "The cap is the lesser of this amount and the value of the unauthorized transfers occurring in the "
            "relevant window; it is a ceiling, not a charge."
        ),
    }


# ---------------------------------------------------------------------------
# Regulation Z
# ---------------------------------------------------------------------------

def regulation_z(
    notice_date: str | date,
    statement_date: str | date,
    *,
    billing_cycle_days: int = 30,
) -> DeadlineResult:
    """Billing error resolution timing under 12 CFR 1026.13.

    Note the ``in no event later than 90 days`` clause: two complete billing
    cycles on a 50-day cycle would be 100 days, and the 90-day cap binds. A
    calculation that applies only the cycle rule produces a deadline ten days
    too late on exactly the accounts where cycles are long.
    """
    notice = parse_date(notice_date)
    statement = parse_date(statement_date)
    result = DeadlineResult(regulation="REG_Z", timely=True)

    notice_deadline = add_calendar_days(statement, 60)
    result.timely = notice <= notice_deadline
    result.citations.append("1026.13(b)")
    result.deadlines.append(Deadline(
        "billing error notice deadline", notice_deadline, "calendar days", 60, "1026.13(b)"))
    if not result.timely:
        result.findings.append(
            f"The billing error notice arrived after the 60-day window closed on {notice_deadline.isoformat()}, "
            "so the 1026.13 resolution procedures are not triggered."
        )
        return result

    result.deadlines.append(Deadline(
        "written acknowledgement to the consumer", add_calendar_days(notice, 30), "calendar days", 30,
        "1026.13(c)(1)", "not required if the dispute is fully resolved within the same 30 days"))
    result.citations.append("1026.13(c)(1)")

    two_cycles = add_calendar_days(notice, 2 * billing_cycle_days)
    ninety = add_calendar_days(notice, 90)
    resolution_due = min(two_cycles, ninety)
    result.deadlines.append(Deadline(
        "resolution complete", resolution_due, "calendar days", (resolution_due - notice).days, "1026.13(c)(2)",
        "two complete billing cycles" if resolution_due == two_cycles else "the 90-day cap binds before two cycles elapse"))
    result.citations.append("1026.13(c)(2)")

    result.findings.append(
        "The creditor may not attempt to collect the disputed amount, report it as delinquent, or close the "
        "account solely because the consumer asserted billing error rights, until the dispute is resolved."
    )
    result.citations.append("1026.13(d)")
    return result


# ---------------------------------------------------------------------------
# FCRA and FDCPA
# ---------------------------------------------------------------------------

def fcra(dispute_date: str | date, *, additional_information_provided: bool = False) -> DeadlineResult:
    dispute = parse_date(dispute_date)
    result = DeadlineResult(regulation="FCRA", timely=True)
    days = 45 if additional_information_provided else 30
    result.deadlines.append(Deadline(
        "reinvestigation complete", add_calendar_days(dispute, days), "calendar days", days, "1681i(a)(1)",
        "extended to 45 days because the consumer supplied further relevant information"
        if additional_information_provided else ""))
    result.deadlines.append(Deadline(
        "notify the consumer of any reinsertion", add_calendar_days(dispute, days), "business days", 5,
        "1681i(a)(5)", "five business days after any reinsertion of deleted information"))
    result.citations.extend(["1681i(a)(1)", "1681i(a)(5)", "1681s-2(b)"])
    result.findings.append(
        "The furnisher must complete its own investigation within the same window and report corrections to every "
        "agency it supplied."
    )
    return result


def fdcpa(first_contact_date: str | date, *, written_dispute_date: str | date | None = None) -> DeadlineResult:
    first_contact = parse_date(first_contact_date)
    result = DeadlineResult(regulation="FDCPA", timely=True)
    result.deadlines.append(Deadline(
        "validation notice sent", add_calendar_days(first_contact, 5), "calendar days", 5, "1692g(a)"))
    result.deadlines.append(Deadline(
        "consumer dispute window closes", add_calendar_days(first_contact, 30), "calendar days", 30, "1692g(a)"))
    result.citations.append("1692g(a)")

    if written_dispute_date is not None:
        dispute = parse_date(written_dispute_date)
        within = (dispute - first_contact).days <= 30
        result.timely = within
        if within:
            result.findings.append(
                f"A written dispute was received on {dispute.isoformat()}, within the 30-day window. Collection must "
                "cease until verification of the debt is obtained and mailed to the consumer."
            )
            result.citations.append("1692g(b)")
        else:
            result.findings.append(
                f"The written dispute was received on {dispute.isoformat()}, after the 30-day validation window. The "
                "cessation requirement is not triggered, though the collector remains bound by the prohibitions on "
                "false representations and harassment."
            )
            result.citations.extend(["1692e", "1692d"])
    return result


DISPATCH = {"REG_E": regulation_e, "REG_Z": regulation_z, "FCRA": fcra, "FDCPA": fdcpa}


def regulation_for_issue(issue: str) -> str:
    return REGULATION_BY_ISSUE.get(issue, "REG_E")
