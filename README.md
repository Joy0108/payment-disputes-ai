# AI System for Payment Disputes and Consumer Financial Complaints

Intake to cited draft response, orchestrated as a **LangGraph** state graph.
Four models classify and score the complaint, a deterministic rules engine
computes every regulatory deadline, section-level retrieval finds the governing
provision, and a generator writes the letter — under a verifier that rejects any
draft citing a section that does not exist or stating a date the rules engine
never produced.

```bash
pip install -e ".[dev,serving]"
make build                 # raw -> bronze -> silver -> gold, validated at each boundary
make train                 # the four models
make eval                  # full evaluation and the promotion gates
disputes deadlines         # 22 hand-computed regulatory deadline cases
disputes workflow          # the compiled LangGraph, as mermaid
disputes serve             # FastAPI on :8000
```

> **Scope.** 9,000 complaints across 20 issue classes, 24,000 fraud
> applications at a 1.19% base rate, 29 sections of Reg E / Reg Z / FCRA / FDCPA
> and CFPB circulars, a 30-question frozen retrieval set, and 22 hand-computed
> deadline cases. Every number below comes from `make eval` on this machine.
> Two of the results are negative and they are in the table with the rest.
> The corpora are synthetic, generated from a seed with two documented
> distribution shifts injected, so every result here is reproducible from a
> clean checkout.

---

## Results

| | metric | value |
|---|---|---|
| **Regulatory deadlines** | hand-computed cases correct | **22 / 22** |
| **Retrieval** | Recall@4 | **0.850** (MRR 0.694) |
| | governing-regulation routing | **0.933** |
| | rules engine agrees with the golden answer | **8 / 8** |
| **Draft verification** | citation resolution | **1.000** (1,066 citations over 120 disputes) |
| | dates not produced by the rules engine | **0** |
| | uncited factual claims | **0** |
| **Issue classification** | macro F1, 20 classes | **0.830** |
| **Escalation risk** | ROC AUC | **0.713** |
| **Outcome prediction** | balanced accuracy, 5 classes | 0.290 (chance 0.200) |
| **Fraud** | recall @ 2% alert budget | 0.134 (precision 0.108, **6.7× lift**, AUC 0.810) |

All seven promotion gates pass. The last two numbers are weak and the gates are
set to match them, not to flatter them — see *Where this is weak* below.

---

## The eight-step workflow

```
intake ─► classify ─┬─► validate ─► compute_deadlines ─► assess_risk ─► retrieve_regulation ─► draft ─► verify ─┐
                    │                                                                            ▲            │
                    │                                                                            └── revise ◄──┤
                    └─► unclassifiable ──────────────────────────────────────────────────────────────────► route
```

**The split between step 4 and step 7 is the whole design.** Everything with a
right answer — which regulation governs, what date the investigation must
conclude, which network reason code applies — is computed deterministically and
handed to the generator as fact. The generator writes prose. Step 8 checks it
did not add anything.

An LLM asked to compute a Regulation E provisional-credit deadline will usually
be right. "Usually" is the wrong standard for a date that determines whether a
bank owes a consumer money, and a model that is wrong 2% of the time about
Thanksgiving is wrong in a way nobody notices until an examiner does.

`compute_deadlines` and `verify` are registered as **required** stages: the
engine raises if a run finishes without them. The `unclassifiable` branch is
registered as an escape, because a complaint the pipeline cannot categorise
produces no letter, and demanding it computed a deadline first would be
demanding it guess.

---

## Orchestration: LangGraph, and a control to prove it

The workflow runs on **LangGraph**. The topology — nodes, edges, routers, the
required-stage rule — is declared once in `workflow/spec.py` and compiled into
a `StateGraph`. Four things made the library worth the dependency, and each one
replaced something this codebase would otherwise maintain by hand:

**Reducers put the merge rule in the type.** `_path` and `_checkpoints` are
`Annotated[list[...], operator.add]` on the state schema, so "a node returns a
partial update that is merged, never substituted" stops being a convention the
engine enforces and becomes a property a node *cannot* violate. It could not
overwrite the audit trail if it tried.

**The checkpointer is the audit trail.** Every super-step is persisted, so
`state_history()` answers the examiner's question — *what did the system know,
and when* — from the framework's own durable record rather than from a list
this code appends to and could forget to append to. State types crossing the
checkpointer are registered explicitly (`ALLOWED_STATE_TYPES`); LangGraph
blocks deserialising arbitrary classes out of a checkpoint, and it is right to.

**Interrupts are the human gate.** The workflow already computes
`requires_human_approval`. With `human_in_the_loop=True` the graph *stops*
before it routes, and `resume()` continues from the persisted checkpoint — so
the approver reads the finished draft and its verification, the thing being
approved, rather than a queue assignment already made on their behalf. A pause
that survives process death is a durability property, and not one worth
hand-writing.

**`recursion_limit` bounds the cycle.** `verify → draft` is a real cycle. A
verifier that never passes terminates with a graph error instead of spinning.

What LangGraph does *not* provide is the required-stage rule — there is no way
to declare "this run is invalid unless `compute_deadlines` executed". That
check stays ours and runs against the accumulated path, honouring the escape.

### The conformance test

`workflow/graph.py` holds a second executor: a dependency-free walker over the
same `WorkflowSpec`. It is not a fallback anyone is expected to run — it is the
**control**.

Both engines call the same router functions and the same `missing_required`
check, so the test asserting that one dispute produces an identical path, audit
trail, letter and routing decision under both is asserting that the two
executors agree — not that two hand-written copies of a graph happen to have
been edited in step. CI runs it as a step of its own:

```
python -m disputes.cli dispute --engine langgraph ... > langgraph.txt
python -m disputes.cli dispute --engine reference ... > reference.txt
diff langgraph.txt reference.txt
```

Reducers, conditional edges and the recursion bound either mean what the
reference walker means, or `diff` says so. That is the difference between using
a framework and understanding it.

---

## The rules engine

Business-day arithmetic under the federal holiday schedule, computed from
5 U.S.C. 6103 rather than tabulated — a table has an expiry date and this code
will run on dates nobody has entered yet.

Every expected value in `eval/deadlines.py` was worked out by hand from the
regulation and a calendar. A test whose expectation came from the
implementation proves the implementation agrees with itself and nothing else.

The cases cover the interactions rather than the happy path:

| case | result | why it is not obvious |
|---|---|---|
| 10 business days from Wed 20 Nov 2024 | 5 Dec | Thanksgiving falls inside the window |
| 3 business days from Fri 27 Dec 2024 | 2 Jan 2025 | 1 January is a holiday |
| Juneteenth in 2020 | not a holiday | federal only from June 2021 |
| Juneteenth 2021 (a Saturday) | observed Fri 18 June | fixed-date holidays shift |
| Reg E, point of sale + provisional credit | 90 days | but the **initial** period stays at 10 business days — the 20-day rule is new-account only |
| Reg E, new account | 20 business days | and 90 days on the outer period |
| Reg E, no provisional credit | no extension at all | the extension is *conditional* on the credit |
| Reg Z, 50-day billing cycles | 90 days, not 100 | "in no event later than 90 days" binds |
| Reg E liability tiers | $50 / $500 / uncapped | tiers 1–2 count **business days from discovery**; tier 3 counts **calendar days from the statement**. Two different clocks. |

Counting starts the day *after* receipt. That off-by-one produces a deadline a
day early, which is a violation rather than a rounding error.

`deadline_exactness` is gated at **1.0**, absolutely. There is no rate at which
some wrong deadlines are acceptable.

---

## Data layers and the splits

`raw → bronze → silver → gold`, with declarative expectations at every
boundary. Each expectation carries the **consequence** of its violation,
because "expected complaint_id to be unique" tells an on-call engineer nothing
about whether to page someone.

Silver **quarantines** rather than drops. A silently shrinking row count is the
hardest data bug to find; `test_silver_quarantines_rather_than_drops` asserts
kept + quarantined equals input.

**Splits are dates, not fractions, and frozen into the row.** A fraction moves
every time a row is added, so two runs a week apart are incomparable and a
regression looks like noise. Both tables assert `monotonic` on their ordering
key — an unordered table turns a temporal split into a random one, silently.

The manifest content-hashes every layer and emits `.dvc` pointers, so a model
version can name the exact bytes it trained on and `disputes verify` makes
"the model is stale" a detectable state rather than a suspicion.

### The sampling bias, documented rather than fixed

68.8% of complaints come from four large institutions. That is a property of
the real CFPB database — those are the firms consumers know how to complain
about — and it is reproduced here rather than balanced away, because balancing
it would hide it. Any per-company conclusion from this data is about complaint
*routing*, not about firm conduct.

---

## Where the temporal split earns its keep

The corpora carry two deliberate distribution shifts, and both are invisible
under a random split:

**Company response policy shifts after 2022-09** — relief rates fall about 10
points. The outcome model scores 0.290 balanced accuracy on the temporal split
against **0.306 on a shuffled split of the same data**, and the shuffled number
is the one a random-split benchmark would have reported for a model that is
worse in production.

**The fraud model has a concept drift**: `keep_alive_session` inverts its
relationship to the label at month 4 — early in the window keeping a session
alive is a legitimate-user habit, later it is automation. A model fitted across
the whole period averages the two regimes, and its coefficient lands at
**−0.018**, near zero. That collapsed coefficient is the signature of a concept
drift, and it is reported in `models.json` rather than left as an unexplained
metric drop.

---

## Retrieval

Chunked by **section**, not by token window. A citation in a consumer letter
has to name a provision the recipient can look up, and a sliding window cutting
across 1005.11(c)(1) and (c)(2) produces a chunk that cites neither while
answering with a deadline drawn from both.

Two lexical bridges, both necessary:

- **Numerals.** Regulations write "sixty days"; consumers write "60 days". The
  substitution runs on the phrase before tokenisation — per token it fails,
  because "twenty-five" has already split into two words by then. Word-bounded,
  so `2025` is untouched.
- **Section identifiers** are indexed by their parts, so a query naming
  `1005.11` finds `1005.11(c)(2)`.

### The ablation: routing is worth more than fusion

| config | Recall@4 | governing-regulation routing |
|---|---|---|
| BM25 only | 0.750 | 0.700 |
| dense only | 0.783 | 0.767 |
| RRF fused | 0.783 | 0.800 |
| **RRF + regulation-aware rerank** | **0.850** | **0.933** |

Fusion buys 0.033 recall. The **regulation-aware rerank buys another 0.067, and
0.133 on routing** — by far the larger win. The reason is specific: "unauthorized
use of a card" retrieves the Reg Z credit-card liability section and the Reg E
debit section about equally well, and answering a debit dispute out of Reg Z is
wrong *on the law*, not merely off-topic. Plain recall scores that as a hit.
Off-regulation sections are demoted rather than removed, because cross-references
between Reg E and Reg Z are real.

---

## Verification: what actually stops a bad letter

Three checks on the generator's output, none of them requested in the prompt:

1. **Citations resolve** to a retrieved section or a computed deadline.
2. **Every date appears in the rules engine's output.** Not "looks plausible" —
   the exact set. A date in a consumer letter that the code never produced is
   the most dangerous thing this system could emit.
3. **Factual sentences carry a citation.**

Failure loops back into drafting up to three attempts. It does not warn and
continue. Over 120 disputes: 1,066 citations, **100% resolved, zero invented
dates, zero uncited claims**, all on the first attempt.

The template generator is deterministic, so those numbers measure the *plumbing
and the verifier* rather than a model's honesty — which is exactly what makes
them a reproducible CI gate. Point `DISPUTES_LLM=anthropic` at the Claude
backend and the same three checks become a real measurement of the model, with
the identical code path.

---

## The curriculum experiment, which failed

The QLoRA fine-tune (`llm/qlora.py`) needs a GPU and does not run in CI. But
the claim underneath a curriculum — that easy-before-hard changes what a model
ends up with — is testable on any model trained by SGD over an ordered stream,
and this project has one. Single pass, identical seed, identical features,
ordering the only difference:

| ordering | macro F1 |
|---|---|
| shuffled | **0.807** |
| easy → hard | 0.769 |
| hard → easy | 0.644 |

**Curriculum ordering costs 0.038 F1.** The diagnosis is measured, not guessed:
difficulty is *not independent of the label*. **42% of its variance is explained
by the issue class** (η² = 0.42), because difficulty is driven by deadline count
and extension branches, which are regulation-specific. Sorting by difficulty
therefore partially sorts by class, and one pass of SGD over class-blocked data
ends fitted to whatever it saw last. The ordering was not teaching an easier
version of the problem first; it was removing the shuffling that prevents
recency bias.

`qlora.py` consequently uses **bucketed** curriculum — five difficulty bands,
shuffled within each — which keeps the easy-first progression and breaks the
class blocking. That is what the negative result was worth.

---

## Drift monitoring, and the baseline that hides drift

Data drift and concept drift are monitored separately because they call for
different responses: inputs moving may be fine, the input-output relationship
moving means the model is wrong and no feature monitor will say so.

The result worth stating:

| baseline | verdict |
|---|---|
| whole training set | **stable** |
| first 12 months of training | **investigate**, concept drift detected (PSI 0.129 on realised outcomes) |

The response policy shifts *inside* the training period, so a baseline built
from all of training already contains both regimes, the histograms overlap, and
PSI reports stable while the model is measurably stale. The reference has to be
a window believed stable, not everything the model was fitted on.
`edges_from` takes bin edges from the baseline only, for the same reason —
recomputing them on the current window makes PSI report zero forever.

Shadow mode segments **before** it totals: a candidate that wins on aggregate
while changing decisions in the high-value segment is a policy change, not a
free swap, and the report says so rather than reporting one agreement number.

---

## Where this is weak

Two numbers are poor and are reported at their real level:

- **Outcome prediction, 0.290 balanced accuracy** against a 0.200 chance floor.
  Company response is genuinely close to unpredictable from a complaint
  narrative, and the response policy drifts underneath the model. This is a
  weak model on a hard target, not a well-tuned one.
- **Fraud recall 0.134 at a 2% alert budget.** AUC is 0.810 and lift over random
  is 6.7×, so the ranking has real signal, but at a 1.19% base rate a 2% budget
  is 120 alerts holding 13 of the 97 frauds in the test period. A linear model on 25 features is the
  floor here, not the ceiling.

The gates are set just under these values deliberately. A gate set at an
aspiration is a gate that gets disabled the first time it fires, and then it
was never a gate.

---

## Deploying the API

`deploy/` holds the real target: two workloads from one image, the worker
autoscaled on queue depth, and the shadow and drift CronJobs.

`api/index.py` and `vercel.json` add a serverless deployment for the
**synchronous** endpoint, which is useful as a live demo. Three things about it
are worth stating rather than discovering:

* **It runs the reference engine, not LangGraph.** Both execute the same
  declared topology and the conformance test asserts they produce identical
  output, so this costs nothing behaviourally and keeps `langchain-core`,
  `langsmith` and the rest out of a size-limited bundle. `requirements.txt` is
  correspondingly slim - numpy, FastAPI, pydantic - because scikit-learn and
  langgraph are both imported lazily and the serving path reaches neither.
* **It serves without trained models.** The estimator artifacts are build
  outputs and are not committed. `/health` reports `models_loaded: false`, and
  the rules engine, validation, retrieval, drafting and the citation verifier
  all work from a stated issue. The issue classifier and risk scores are what
  is missing.
* **`POST /disputes/batch` is not durable there.** The queue writes to `/tmp`,
  which is per-invocation and not shared between instances, so a job enqueued
  by one request is invisible to the next. The queue is designed for the
  long-lived worker in `deploy/`; on serverless, use `POST /disputes`.

Training, the cron jobs and the batch worker have no serverless equivalent.
Anything beyond the synchronous demo wants the Kubernetes manifests, or a host
with a persistent disk and a background process.

---

## Commands

```
disputes build [--lenient]     bronze, silver, gold, with contracts at each boundary
disputes verify                re-hash the tracked data and report what moved
disputes train                 the four models, TPE hyperparameter search
disputes deadlines             the 22 hand-computed regulatory cases
disputes dispute [--engine]    the eight-step workflow on one dispute
disputes regulations "<q>"     query the regulation corpus
disputes curriculum            build the SFT dataset and run the ordering experiment
disputes qlora-plan            the fine-tune plan, without a GPU
disputes shadow                shadow-mode comparison, segmented
disputes drift                 drift monitors against a reference window
disputes eval [--fast]         full evaluation and the promotion gates
disputes serve                 FastAPI on :8000
disputes workflow [--engine]   print the compiled graph as mermaid (LangGraph's own renderer)
```

## Layout

```
scripts/build_data.py     seeded generators for both corpora
scripts/narratives.py     narrative synthesis, with the leakage note that motivated it
data/regulations/         Reg E, Reg Z, FCRA, FDCPA, CFPB circulars at section level
data/golden/              the frozen 30-question retrieval set
src/disputes/rules/       calendar, deadlines, validation, reason codes
src/disputes/data/        layers, declarative contracts, content-addressed manifest
src/disputes/models/      TPE search and the four models
src/disputes/rag/         section retrieval, drafting, the verifier
src/disputes/llm/         SFT dataset, curriculum experiment, QLoRA config
src/disputes/workflow/    spec (topology), the LangGraph engine, the reference walker, the eight steps
src/disputes/monitoring/  drift and shadow mode
src/disputes/service/     FastAPI and the durable batch queue
deploy/                   Kubernetes: split API/worker scaling, shadow and drift CronJobs
tests/                    96 tests, including the cross-engine conformance test; `make test`
```

## License

MIT.
