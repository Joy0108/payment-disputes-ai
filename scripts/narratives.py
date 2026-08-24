"""Narrative synthesis for the complaint corpus.

Split out from the generator because getting this wrong makes the whole
benchmark worthless, and the first version did.

The first attempt gave every issue its own private set of sentences. The result
was a classification task solvable at macro F1 1.000 by a bag of words - the
label was recoverable from a single token. A benchmark that reports a perfect
score is not measuring a model, it is measuring how leaky the generator is.

What replaces it:

* a **shared pool** of ~40 sentences that appear across many issues and carry
  no label information;
* a small number of **signature** sentences per issue, of which a narrative
  draws only one or two, and which are themselves shared between *related*
  issues - the ones a human annotator would also confuse;
* **confusion groups**, so that the errors the classifier makes are the errors
  the taxonomy invites: unauthorized-transaction against fraud-or-scam, the two
  credit-reporting issues against each other, the two debt-collection ones.

That produces a task with real signal and an irreducible error floor, which is
what the real CFPB free-text field looks like.
"""

from __future__ import annotations

import random

# Sentences that appear in complaints about anything. No label information.
SHARED = [
    "I contacted the company on {sdate} and got nowhere",
    "I have called {n} times about this",
    "nobody would escalate my case",
    "I am attaching the statement showing the entry",
    "this has been going on since {sdate}",
    "I am asking for {amount} to be returned to me",
    "the representative put me on hold for over an hour",
    "I was transferred between {n} departments",
    "I have sent documents twice and they say they never arrived",
    "the company closed my case without telling me",
    "I received a letter that did not address what I asked",
    "I am a customer of {n} years and this is how I am treated",
    "I want this resolved before it affects my credit",
    "the amount involved is {amount}",
    "I have been given a different answer every time I call",
    "I asked for a supervisor and was refused",
    "the online portal does not show the case at all",
    "I filed this complaint because the company stopped responding",
    "my account has been affected since {sdate}",
    "I need this corrected in writing",
    "the company says it is following its policy but will not show me the policy",
    "I have kept records of every call",
    "this is causing me real financial hardship",
    "I am asking the regulator to look at this",
    "the response I received was a form letter",
    "I was told someone would call back and nobody did",
    "the fee involved was {amount}",
    "I have disputed this {n} times now",
    "the company acknowledged the problem on the phone but not in writing",
    "I want an explanation of how the amount was calculated",
]

# Signature sentences. The list per issue is short, a narrative draws one or two,
# and several are deliberately shared with a confusable neighbour.
SIGNATURE = {
    "Unauthorized transactions or other transaction problem": [
        "there are charges on my account I did not authorize",
        "someone used my debit card without my permission",
        "provisional credit was never applied while the investigation ran",
        "the bank said the transaction was authorized because a PIN was used",
    ],
    "Fraud or scam": [
        "there are charges on my account I did not authorize",
        "I was tricked into sending a payment by someone pretending to be the bank",
        "the transfer could not be recalled once it left my account",
        "I reported the scam the same day and the funds were already gone",
    ],
    "Problem with a purchase shown on your statement": [
        "the merchant billed me for goods that never arrived",
        "I disputed the billing error in writing",
        "the item went back but the credit never appeared",
        "I was billed twice for the same purchase",
    ],
    "Charged fees or interest you didn't expect": [
        "I was charged interest despite paying the balance in full",
        "a fee appeared that no disclosure mentioned",
        "the promotional rate ended earlier than I was told",
        "I was billed twice for the same purchase",
    ],
    "Other features, terms, or problems": [
        "the terms of my account changed without notice",
        "my credit limit was cut with no explanation",
        "a benefit I was promised at application does not exist",
    ],
    "Getting a credit card": [
        "my application was declined and I got no adverse action notice",
        "the approval terms are not the terms that were advertised",
        "an application was submitted in my name that I did not make",
    ],
    "Incorrect information on your report": [
        "an account on my credit report is not mine",
        "the balance being reported is wrong",
        "the account was paid but reports as charged off",
    ],
    "Problem with a company's investigation into an existing problem": [
        "the furnisher verified the item without looking at my documents",
        "an account on my credit report is not mine",
        "I received a form letter saying the item was verified",
        "the same error came back after it was deleted",
    ],
    "Improper use of your report": [
        "a company pulled my credit report with no permissible purpose",
        "there is a hard inquiry I never authorised",
        "my report was accessed by a company I have no relationship with",
    ],
    "Attempts to collect debt not owed": [
        "the collector is chasing a debt that was already paid",
        "this debt belongs to somebody else with a similar name",
        "they could not produce validation of the debt",
    ],
    "Written notification about debt": [
        "I never received a validation notice after the first contact",
        "the notice did not state the amount of the debt",
        "they could not produce validation of the debt",
    ],
    "Communication tactics": [
        "the collector called me repeatedly in a single day",
        "they contacted me at work after I told them to stop",
        "the caller threatened action they could not take",
    ],
    "Managing an account": [
        "the bank changed my account terms without notice",
        "I could not get into online banking for days",
        "my direct deposit went to the wrong account",
    ],
    "Closing an account": [
        "the bank closed my account and would not say why",
        "my remaining balance was never returned",
        "the account was closed while a dispute was still open",
    ],
    "Problem caused by your funds being low": [
        "I was charged several overdraft fees in one day",
        "the bank reordered my transactions largest to smallest",
        "an overdraft fee hit a transaction authorized on a positive balance",
    ],
    "Trouble using your card": [
        "my prepaid card was declined although it had a balance",
        "the card was locked and nobody could tell me why",
        "the replacement card never arrived",
    ],
    "Problem with a lender or other company charging your account": [
        "a company debited my account after I revoked authorisation",
        "the recurring payment kept coming after I cancelled",
        "the amount taken was more than I authorised",
    ],
    "Money was not available when promised": [
        "the transfer was promised in minutes and took days",
        "the recipient never got the funds and I was not refunded",
        "the company held the transfer for a review with no update",
    ],
    "Confusing or missing disclosures": [
        "the fee schedule was never given to me before the account opened",
        "the terms on the website differ from the agreement",
        "I was not given a change in terms notice",
    ],
    "Struggling to repay your loan": [
        "the lender would not discuss a repayment plan",
        "the fees now exceed what I originally borrowed",
        "payments kept being taken after I asked them to stop",
    ],
}

# Issues a human annotator confuses. Cross-contamination is drawn from inside
# the group most of the time, so the error structure is realistic rather than
# uniform noise.
CONFUSION_GROUPS = [
    ["Unauthorized transactions or other transaction problem", "Fraud or scam",
     "Problem with a lender or other company charging your account"],
    ["Problem with a purchase shown on your statement", "Charged fees or interest you didn't expect",
     "Other features, terms, or problems"],
    ["Incorrect information on your report", "Problem with a company's investigation into an existing problem",
     "Improper use of your report"],
    ["Attempts to collect debt not owed", "Written notification about debt", "Communication tactics"],
    ["Managing an account", "Closing an account", "Confusing or missing disclosures"],
    ["Money was not available when promised", "Trouble using your card"],
]

_GROUP_OF = {issue: group for group in CONFUSION_GROUPS for issue in group}


def compose(issue: str, rng: random.Random, amount: float, sdate: str) -> str:
    """One narrative. Roughly a third of its sentences carry label information."""
    parts: list[str] = []

    signature = SIGNATURE.get(issue, [])
    if signature:
        parts.extend(rng.sample(signature, k=min(rng.choice([1, 1, 2]), len(signature))))

    parts.extend(rng.sample(SHARED, k=rng.randint(2, 4)))

    # Contamination: a sentence from a neighbour the taxonomy invites confusion
    # with, or occasionally from anywhere at all.
    if rng.random() < 0.30:
        group = _GROUP_OF.get(issue, [])
        pool = [i for i in group if i != issue] if (group and rng.random() < 0.75) else [
            i for i in SIGNATURE if i != issue
        ]
        if pool:
            other = rng.choice(pool)
            parts.append(rng.choice(SIGNATURE[other]))

    # A fraction of complaints are almost pure boilerplate, which is where the
    # irreducible error lives.
    if rng.random() < 0.08:
        parts = rng.sample(SHARED, k=rng.randint(3, 5))

    rng.shuffle(parts)
    text = ". ".join(part.rstrip(".") for part in parts) + "."
    return text.format(
        n=rng.randint(2, 9),
        amount=f"${amount:,.2f}",
        sdate=sdate,
    )
