"""
Assembles the LangGraph StateGraph.

Routing summary:

    triage --(classification)--> retrieval | clarification | escalation | safe_response
    retrieval -> generation -> verification
    verification --(pass)--> END
    verification --(fail, retry_count < MAX)--> generation   [loop guard: retry_count]
    verification --(fail, retry_count >= MAX)--> END          [state.classification -> "safe_failure"]
    clarification | escalation | safe_response -> END

The retry_count check inside verification_node + route_after_verification is
the infinite-loop guard: LangGraph's own recursion_limit (set at invoke time)
is a second, coarser safety net in case of a bug in that logic.
"""
from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from src.models import Generator
from src.nodes import (
    clarification_node,
    escalation_node,
    generation_node,
    retrieval_node,
    route_after_triage,
    route_after_verification,
    safe_response_node,
    triage_node,
    verification_node,
)
from src.retrieval import Retriever
from src.state import SupportState


def build_graph(generator: Generator, retriever: Retriever):
    graph = StateGraph(SupportState)

    graph.add_node("triage", partial(triage_node, generator=generator))
    graph.add_node("retrieval", partial(retrieval_node, retriever=retriever))
    graph.add_node("generation", partial(generation_node, generator=generator))
    graph.add_node("verification", verification_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("safe_response", safe_response_node)

    graph.set_entry_point("triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "retrieval": "retrieval",
            "clarification": "clarification",
            "escalation": "escalation",
            "safe_response": "safe_response",
        },
    )

    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", "verification")

    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "retry": "generation",
            "end": END,
        },
    )

    graph.add_edge("clarification", END)
    graph.add_edge("escalation", END)
    graph.add_edge("safe_response", END)

    return graph.compile()


def run_query(query: str, generator: Generator, retriever: Retriever) -> dict:
    """Convenience entry point: builds initial state, runs the graph, returns final state."""
    app = build_graph(generator, retriever)
    initial_state: SupportState = {
        "query": query,
        "retry_count": 0,
        "node_trace": [],
        "warnings": [],
    }
    # recursion_limit is a coarse second-layer loop guard on top of retry_count
    final_state = app.invoke(initial_state, config={"recursion_limit": 15})
    return final_state
