"""The dispute workflow executed on LangGraph.

This is the production engine. It compiles the shared ``WorkflowSpec`` into a
``langgraph.graph.StateGraph`` and runs it under a checkpointer, which is the
reason the library is here rather than the hand-rolled walker in ``graph.py``:

* **Reducers express the merge rule in the type.** ``_path`` and
  ``_checkpoints`` are ``Annotated[list, operator.add]``, so "a node returns a
  partial update that is merged, never substituted" stops being a convention
  the engine enforces and becomes a property of the state schema. A node that
  tried to overwrite the audit trail could not.

* **The checkpointer is the audit trail.** Every super-step is persisted, so
  ``state_history()`` answers an examiner's question - *what did the system
  know, and when* - from the framework's own record rather than from a list
  this codebase maintains and could forget to append to.

* **Interrupts are the human gate.** The workflow already computes
  ``requires_human_approval``. With ``human_in_the_loop=True`` the graph
  actually stops before it routes, and ``resume()`` continues from the
  persisted checkpoint. A pause that survives process death is a durability
  property, and it is not one worth hand-writing.

* **The cycle bound is the framework's.** ``verify -> draft`` is a real cycle;
  ``recursion_limit`` bounds it, so a verifier that never passes terminates
  with a graph error instead of looping.

What LangGraph does *not* give us is the required-stage rule: there is no way
to declare "this run is invalid unless ``compute_deadlines`` executed". That
check stays ours and runs after ``invoke``, against the accumulated path.
"""

from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, TypedDict

from .spec import END, START, WorkflowError, WorkflowSpec


def _assert_sentinels_match() -> None:
    """Our START/END are LangGraph's, so a mapping value passes through unchanged.

    Asserted rather than assumed: if a future LangGraph renames them, an edge
    to ``__end__`` would quietly become an edge to a node that does not exist.
    """
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START

    if (START, END) != (LG_START, LG_END):
        raise WorkflowError(
            f"sentinel mismatch: spec uses {(START, END)}, langgraph uses {(LG_START, LG_END)}"
        )

# The state objects the checkpointer has to round-trip. Registering them is
# deliberate: LangGraph blocks deserialising arbitrary types out of a
# checkpoint, and it is right to. A workflow whose state can carry anything is
# a deserialisation gadget.
ALLOWED_STATE_TYPES = [
    ("disputes.rag.draft", "Draft"),
    ("disputes.rag.draft", "Verification"),
    ("disputes.rag.index", "Section"),
]


class DisputeState(TypedDict, total=False):
    """The workflow state.

    Everything except the two accumulators is last-write-wins, which is what
    the nodes expect: ``draft`` replaces the previous draft on a revision,
    ``attempts`` counts up. The accumulators append, so the record of what ran
    is additive by construction.
    """

    complaint: dict[str, Any]
    prediction: dict[str, Any]
    issue: str | None
    regulation: str | None
    validation: dict[str, Any]
    reason_code: dict[str, Any]
    deadlines: dict[str, Any]
    liability: dict[str, Any] | None
    risk: dict[str, Any]
    sections: list[Any]
    draft: Any
    verification: Any
    critique: Any
    attempts: int
    outcome: dict[str, Any]
    error: str
    _path: Annotated[list[str], operator.add]
    _checkpoints: Annotated[list[dict[str, Any]], operator.add]


def _checkpointer() -> Any:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES))


def _instrument(name: str, spec: WorkflowSpec):
    """Wrap a node so it records its own transition into the state.

    The wrapper is what keeps the two engines' audit trails comparable: it
    records the same fields the reference walker records, and it resolves the
    successor through ``spec.next_after`` - the very function LangGraph's
    conditional edge will call a moment later to make the same decision.
    """
    fn = spec.nodes[name]

    def node(state: DisputeState) -> dict[str, Any]:
        started = time.perf_counter()
        before = set(state)
        update = fn(dict(state)) or {}
        if not isinstance(update, Mapping):
            raise WorkflowError(f"node {name!r} returned {type(update).__name__}, expected a mapping")
        update = dict(update)
        checkpoint = {
            "step": len(state.get("_path", [])),
            "node": name,
            "next": spec.next_after(name, {**state, **update}),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "added": sorted(set(update) - before),
        }
        return {**update, "_path": [name], "_checkpoints": [checkpoint]}

    return node


class LangGraphWorkflow:
    """A compiled LangGraph app behind the same interface as ``StateMachine``."""

    engine = "langgraph"

    def __init__(self, spec: WorkflowSpec, human_in_the_loop: bool = False):
        from langgraph.graph import StateGraph

        _assert_sentinels_match()
        spec.validate()
        self.spec = spec
        self.name = spec.name
        self.human_in_the_loop = human_in_the_loop

        builder = StateGraph(DisputeState)
        for node_name in spec.nodes:
            builder.add_node(node_name, _instrument(node_name, spec))
        builder.add_edge(START, spec.entry)
        for src, dst in spec.edges.items():
            builder.add_edge(src, dst)
        for src, (router, mapping) in spec.conditional.items():
            builder.add_conditional_edges(src, router, mapping)

        # The gate pauses *before* routing, so the state an approver reads is
        # the finished draft and its verification - the thing being approved -
        # rather than a queue assignment already made on their behalf.
        interrupt_before = ["route"] if human_in_the_loop and "route" in spec.nodes else []
        self.app = builder.compile(checkpointer=_checkpointer(), interrupt_before=interrupt_before)

    # -- execution ---------------------------------------------------------
    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.spec.max_steps}

    def _finish(self, result: Mapping[str, Any], thread_id: str, *, enforce: bool) -> dict[str, Any]:
        state = dict(result)
        state.setdefault("_path", [])
        state.setdefault("_checkpoints", [])
        state["_thread_id"] = thread_id
        state["_engine"] = self.engine
        state["_steps"] = len(state["_path"])
        if enforce:
            missing = self.spec.missing_required(state["_path"])
            if missing:
                raise WorkflowError(f"required stage(s) did not run: {missing}; path={state['_path']}")
        return state

    def invoke(self, state: Mapping[str, Any], thread_id: str | None = None) -> dict[str, Any]:
        thread_id = thread_id or self._thread_id_for(state)
        try:
            result = self.app.invoke(dict(state), config=self._config(thread_id))
        except Exception as exc:  # recursion limit is the cycle bound
            if type(exc).__name__ == "GraphRecursionError":
                raise WorkflowError(
                    f"{self.name} exceeded max_steps={self.spec.max_steps}; the verify/draft cycle did not settle"
                ) from exc
            raise
        # An interrupted run has not finished, so the required-stage rule does
        # not apply to it yet. It applies to what `resume` returns.
        return self._finish(result, thread_id, enforce=not self.interrupted(thread_id))

    def resume(self, thread_id: str, update: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Continue a run paused at the human gate, optionally amending state."""
        config = self._config(thread_id)
        if update:
            self.app.update_state(config, dict(update))
        result = self.app.invoke(None, config=config)
        return self._finish(result, thread_id, enforce=True)

    def interrupted(self, thread_id: str) -> bool:
        return bool(self.app.get_state(self._config(thread_id)).next)

    def stream(self, state: Mapping[str, Any], thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Per-node updates as they happen, for a progress view or a trace."""
        thread_id = thread_id or self._thread_id_for(state)
        yield from self.app.stream(dict(state), config=self._config(thread_id), stream_mode="updates")

    # -- introspection -----------------------------------------------------
    def state_history(self, thread_id: str) -> list[dict[str, Any]]:
        """The framework's own checkpoint record, oldest first.

        One entry per super-step: which stage had just completed, what was
        pending next, and the path so far. This is the durable answer to *what
        did the system know, and when* - persisted by the checkpointer as the
        run proceeds, so it survives the process that produced it.
        """
        history = list(self.app.get_state_history(self._config(thread_id)))
        entries = []
        for snapshot in reversed(history):
            path = list(snapshot.values.get("_path", []))
            entries.append({
                "step": snapshot.metadata.get("step") if snapshot.metadata else None,
                "completed": path[-1] if path else None,
                "next": list(snapshot.next),
                "path": path,
            })
        return entries

    def to_mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()

    def validate(self) -> None:
        self.spec.validate()

    @staticmethod
    def _thread_id_for(state: Mapping[str, Any]) -> str:
        complaint = state.get("complaint") or {}
        base = complaint.get("complaint_id") or "unassigned"
        return f"{base}:{uuid.uuid4().hex[:8]}"


def langgraph_available() -> bool:
    try:
        import langgraph.graph  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["ALLOWED_STATE_TYPES", "DisputeState", "LangGraphWorkflow", "langgraph_available"]
