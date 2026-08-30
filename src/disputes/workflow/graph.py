"""The reference engine: the same graph, walked without a dependency.

LangGraph is the production executor (``langgraph_engine.py``). This walker
exists for two reasons, and neither is nostalgia:

1. **It keeps the default install small.** ``pip install disputes`` and the
   deadline rules, the data layers and the models all work with numpy and
   scikit-learn. The orchestration stack is a ``[graph]`` extra.
2. **It is the control in the conformance test.** Two independent executors
   over one declared topology, asserted to produce the same path, the same
   draft and the same routing decision, is a stronger statement about the
   workflow than either engine passing its own tests.

Nodes are ``state -> partial_state``, the return value is merged rather than
substituted, edges are declared, and every transition is checkpointed. When a
consumer or an examiner asks why a letter said what it said, the answer is the
checkpoint list: which model produced which prediction, which sections were
retrieved, which deadline the rules engine computed, and whether the verifier
passed on the first attempt or the third.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .spec import END, START, NodeFn, RouterFn, WorkflowError, WorkflowSpec

__all__ = [
    "END",
    "START",
    "Checkpoint",
    "NodeFn",
    "RouterFn",
    "StateMachine",
    "WorkflowError",
    "WorkflowSpec",
    "audit_trail",
]


@dataclass
class Checkpoint:
    step: int
    node: str
    next_node: str
    duration_ms: float
    added: list[str]
    snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "node": self.node, "next": self.next_node,
                "duration_ms": self.duration_ms, "added": self.added}


class StateMachine:
    """Walks a :class:`WorkflowSpec` directly."""

    engine = "reference"

    def __init__(self, name: str, max_steps: int = 40, keep_snapshots: bool = False,
                 spec: WorkflowSpec | None = None):
        self.spec = spec or WorkflowSpec(name=name, max_steps=max_steps)
        self.name = self.spec.name
        self.max_steps = self.spec.max_steps
        self.keep_snapshots = keep_snapshots

    @classmethod
    def from_spec(cls, spec: WorkflowSpec, keep_snapshots: bool = False) -> StateMachine:
        return cls(spec.name, spec.max_steps, keep_snapshots, spec=spec)

    # -- construction (delegates; the topology has one home) ---------------
    def add_node(self, name: str, fn: NodeFn, required: bool = False) -> StateMachine:
        self.spec.add_node(name, fn, required)
        return self

    def set_escape_nodes(self, names: set[str]) -> StateMachine:
        self.spec.set_escape_nodes(names)
        return self

    def add_edge(self, src: str, dst: str) -> StateMachine:
        self.spec.add_edge(src, dst)
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> StateMachine:
        self.spec.add_conditional_edges(src, router, mapping)
        return self

    def validate(self) -> None:
        self.spec.validate()

    # -- execution ---------------------------------------------------------
    def invoke(self, state: Mapping[str, Any], thread_id: str | None = None) -> dict[str, Any]:
        self.validate()
        state = dict(state)
        state.setdefault("_checkpoints", [])
        state.setdefault("_path", [])
        node = self.spec.entry
        assert node is not None
        steps = 0

        while node != END:
            if steps >= self.max_steps:
                raise WorkflowError(f"{self.name} exceeded max_steps={self.max_steps}; path={state['_path']}")
            fn = self.spec.nodes.get(node)
            if fn is None:
                raise WorkflowError(f"unknown node {node!r}")

            started = time.perf_counter()
            before = set(state)
            update = fn(state) or {}
            if not isinstance(update, Mapping):
                raise WorkflowError(f"node {node!r} returned {type(update).__name__}, expected a mapping")
            state.update(update)

            nxt = self.spec.next_after(node, state)
            state["_path"].append(node)
            state["_checkpoints"].append(Checkpoint(
                step=steps, node=node, next_node=nxt,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                added=sorted(set(update) - before),
                snapshot=copy.deepcopy({k: v for k, v in state.items() if not k.startswith("_")})
                if self.keep_snapshots else {},
            ))
            node = nxt
            steps += 1

        missing = self.spec.missing_required(state["_path"])
        if missing:
            raise WorkflowError(f"required stage(s) did not run: {missing}; path={state['_path']}")
        state["_steps"] = steps
        state["_engine"] = self.engine
        state["_thread_id"] = thread_id
        return state

    def to_mermaid(self) -> str:
        return self.spec.to_mermaid()


def audit_trail(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The path from intake to letter, as an examiner would read it.

    Accepts either engine's checkpoints: the walker records
    :class:`Checkpoint` objects, LangGraph's instrumented nodes record plain
    dicts so the checkpointer can serialise them.
    """
    trail = []
    for entry in state.get("_checkpoints", []):
        trail.append(entry.to_dict() if isinstance(entry, Checkpoint) else dict(entry))
    return trail
