from functools import lru_cache
from stock_analyst.graph.builder import compile_graph


@lru_cache(maxsize=1)
def get_graph():
    """Compiles and caches the LangGraph graph — called once at startup."""
    return compile_graph()
