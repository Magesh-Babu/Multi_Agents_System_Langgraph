from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from stock_analyst.graph.state import MEMBERS
from stock_analyst.graph.nodes import (
    router_node,
    fundamental_node,
    sentiment_node,
    technical_node,
    risk_assessment_node,
    real_estate_node,
    final_node,
    condition,
)


def compile_graph():
    """Builds and compiles the multi-agent LangGraph state graph.

    Returns:
        CompiledStateGraph: A compiled graph ready for invocation.
    """
    memory = MemorySaver()
    builder = StateGraph(MessagesState)

    builder.add_node("Router_Agent", router_node)
    builder.add_node("Fundamental_Analysis_Agent", fundamental_node)
    builder.add_node("Sentiment_Analysis_Agent", sentiment_node)
    builder.add_node("Technical_Analysis_Agent", technical_node)
    builder.add_node("Risk_Assessment_Agent", risk_assessment_node)
    builder.add_node("Real_Estate_Agent", real_estate_node)
    builder.add_node("Final_Aggregator_Agent", final_node)

    builder.add_edge(START, "Router_Agent")
    builder.add_conditional_edges("Router_Agent", condition, MEMBERS)
    builder.add_edge("Fundamental_Analysis_Agent", "Final_Aggregator_Agent")
    builder.add_edge("Sentiment_Analysis_Agent", "Final_Aggregator_Agent")
    builder.add_edge("Technical_Analysis_Agent", "Final_Aggregator_Agent")
    builder.add_edge("Risk_Assessment_Agent", "Final_Aggregator_Agent")
    builder.add_edge("Real_Estate_Agent", "Final_Aggregator_Agent")
    builder.add_edge("Final_Aggregator_Agent", END)

    return builder.compile(checkpointer=memory)
