"""Checkpointed state machine.

Same engine idea as any graph orchestrator: nodes are ``state -> partial_state``,
the return value is merged rather than substituted, edges are declared, and
every transition is checkpointed.

The property that matters for a regulated workflow is the last one. When a
consumer or an examiner asks why a letter said what it said, the answer is the
checkpoint list: which model produced which prediction, which sections were
retrieved, which deadline the rules engine computed, and whether the verifier
passed on the first attempt or the third.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

START = "__start__"
END = "__end__"

NodeFn = Callable[[dict[str, Any]], Mapping[str, Any] | None]
RouterFn = Callable[[dict[str, Any]], str]


class WorkflowError(RuntimeError):
    pass


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
    def __init__(self, name: str, max_steps: int = 40, keep_snapshots: bool = False):
        self.name = name
        self.max_steps = max_steps
        self.keep_snapshots = keep_snapshots
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self._entry: str | None = None
        self._required: list[str] = []
        self._escapes: set[str] = set()

    def add_node(self, name: str, fn: NodeFn, required: bool = False) -> StateMachine:
        if name in {START, END}:
            raise WorkflowError(f"{name} is reserved")
        if name in self._nodes:
            raise WorkflowError(f"duplicate node {name!r}")
        self._nodes[name] = fn
        if required:
            self._required.append(name)
        return self

    def set_escape_nodes(self, names: set[str]) -> StateMachine:
        """Nodes whose execution waives the required-stage check.

        "Required" means required on any path that produces an output, not on
        every path. A complaint the classifier cannot categorise never reaches
        the deadline computation, and that is correct behaviour rather than a
        control failure - it is routed to a human precisely because the
        pipeline could not do its job.
        """
        self._escapes = set(names)
        return self

    def add_edge(self, src: str, dst: str) -> StateMachine:
        if src == START:
            self._entry = dst
        else:
            self._edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> StateMachine:
        self._conditional[src] = (router, mapping)
        return self

    def validate(self) -> None:
        if self._entry is None:
            raise WorkflowError("no entry point")
        known = set(self._nodes) | {END}
        for src, dst in self._edges.items():
            if src not in self._nodes or dst not in known:
                raise WorkflowError(f"bad edge {src} -> {dst}")
        for src, (_r, mapping) in self._conditional.items():
            if src not in self._nodes:
                raise WorkflowError(f"conditional edge from unknown node {src}")
            for dst in mapping.values():
                if dst not in known:
                    raise WorkflowError(f"conditional edge to unknown node {dst}")

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.validate()
        state = dict(state)
        state.setdefault("_checkpoints", [])
        state.setdefault("_path", [])
        node = self._entry
        assert node is not None
        steps = 0

        while node != END:
            if steps >= self.max_steps:
                raise WorkflowError(f"{self.name} exceeded max_steps={self.max_steps}; path={state['_path']}")
            fn = self._nodes.get(node)
            if fn is None:
                raise WorkflowError(f"unknown node {node!r}")

            started = time.perf_counter()
            before = set(state)
            update = fn(state) or {}
            if not isinstance(update, Mapping):
                raise WorkflowError(f"node {node!r} returned {type(update).__name__}, expected a mapping")
            state.update(update)

            nxt = self._next(node, state)
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

        escaped = bool(self._escapes & set(state["_path"]))
        missing = [] if escaped else [n for n in self._required if n not in state["_path"]]
        if missing:
            raise WorkflowError(f"required stage(s) did not run: {missing}; path={state['_path']}")
        state["_steps"] = steps
        return state

    def _next(self, node: str, state: dict[str, Any]) -> str:
        if node in self._conditional:
            router, mapping = self._conditional[node]
            key = router(state)
            if key not in mapping:
                raise WorkflowError(f"router for {node!r} returned unmapped key {key!r}")
            return mapping[key]
        return self._edges.get(node, END)

    def to_mermaid(self) -> str:
        lines = ["graph TD", f"    {START}([start]) --> {self._entry}"]
        for src, dst in self._edges.items():
            lines.append(f"    {src} --> {dst}")
        for src, (_r, mapping) in self._conditional.items():
            for key, dst in mapping.items():
                lines.append(f"    {src} -->|{key}| {dst}")
        return "\n".join(lines)


def audit_trail(state: dict[str, Any]) -> list[dict[str, Any]]:
    """The path from intake to letter, as an examiner would read it."""
    return [c.to_dict() for c in state.get("_checkpoints", [])]
