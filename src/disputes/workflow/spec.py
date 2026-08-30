"""The workflow topology, declared once and executed by either engine.

The eight steps, their edges, the routers and the required-stage rule live
here and nowhere else. ``LangGraphWorkflow`` compiles this into a real
``langgraph.graph.StateGraph``; ``StateMachine`` walks it directly with no
third-party dependency.

Declaring the topology separately from its execution is what makes the
conformance test in the suite meaningful. Both engines call *the same* router
functions and *the same* ``missing_required`` check, so when the test asserts
that a dispute produces an identical path, draft and outcome under both, it is
asserting that the two executors agree - not that two hand-written copies of
the graph happen to have been edited in step.

The sentinel names match LangGraph's own (``__start__`` / ``__end__``) so a
mapping value can be handed to either engine unchanged.
"""

from __future__ import annotations

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
class WorkflowSpec:
    """A declared graph: nodes, edges, routers, and the stages that must run."""

    name: str
    entry: str | None = None
    nodes: dict[str, NodeFn] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    edges: dict[str, str] = field(default_factory=dict)
    conditional: dict[str, tuple[RouterFn, dict[str, str]]] = field(default_factory=dict)
    escapes: set[str] = field(default_factory=set)
    max_steps: int = 40

    # -- construction ------------------------------------------------------
    def add_node(self, name: str, fn: NodeFn, required: bool = False) -> WorkflowSpec:
        if name in {START, END}:
            raise WorkflowError(f"{name} is reserved")
        if name in self.nodes:
            raise WorkflowError(f"duplicate node {name!r}")
        self.nodes[name] = fn
        if required:
            self.required.append(name)
        return self

    def add_edge(self, src: str, dst: str) -> WorkflowSpec:
        if src == START:
            self.entry = dst
        else:
            self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> WorkflowSpec:
        self.conditional[src] = (router, mapping)
        return self

    def set_escape_nodes(self, names: set[str]) -> WorkflowSpec:
        """Nodes whose execution waives the required-stage check.

        "Required" means required on any path that produces an output, not on
        every path. A complaint the classifier cannot categorise never reaches
        the deadline computation, and that is correct behaviour rather than a
        control failure - it is routed to a human precisely because the
        pipeline could not do its job.
        """
        self.escapes = set(names)
        return self

    # -- shared semantics --------------------------------------------------
    def validate(self) -> None:
        if self.entry is None:
            raise WorkflowError("no entry point")
        known = set(self.nodes) | {END}
        for src, dst in self.edges.items():
            if src not in self.nodes or dst not in known:
                raise WorkflowError(f"bad edge {src} -> {dst}")
        for src, (_router, mapping) in self.conditional.items():
            if src not in self.nodes:
                raise WorkflowError(f"conditional edge from unknown node {src}")
            for dst in mapping.values():
                if dst not in known:
                    raise WorkflowError(f"conditional edge to unknown node {dst}")

    def next_after(self, node: str, state: Mapping[str, Any]) -> str:
        """Where control goes after ``node``, given the post-update state.

        Both engines route through this. The routers are pure functions of
        state, so LangGraph calling one to pick an edge and this calling the
        same one to record the transition in the audit trail cannot disagree.
        """
        if node in self.conditional:
            router, mapping = self.conditional[node]
            key = router(dict(state))
            if key not in mapping:
                raise WorkflowError(f"router for {node!r} returned unmapped key {key!r}")
            return mapping[key]
        return self.edges.get(node, END)

    def missing_required(self, path: list[str]) -> list[str]:
        """Required stages absent from a finished run, honouring the escapes."""
        if set(self.escapes) & set(path):
            return []
        return [n for n in self.required if n not in path]

    def to_mermaid(self) -> str:
        lines = ["graph TD", f"    {START}([start]) --> {self.entry}"]
        for src, dst in self.edges.items():
            lines.append(f"    {src} --> {dst}")
        for src, (_router, mapping) in self.conditional.items():
            for key, dst in mapping.items():
                lines.append(f"    {src} -->|{key}| {dst}")
        return "\n".join(lines)


__all__ = ["END", "START", "NodeFn", "RouterFn", "WorkflowError", "WorkflowSpec"]
