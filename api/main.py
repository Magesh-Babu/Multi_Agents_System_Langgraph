import json
from typing import AsyncIterator

from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse

from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.dependencies import get_graph

app = FastAPI(
    title="Stock Analyst — Multi-Agent API",
    description="LangGraph-powered multi-agent system for financial and real estate analysis.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, graph=Depends(get_graph)):
    """
    Run the multi-agent graph on a query and return the final answer.
    Blocks until all agents complete and the Final Aggregator responds.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    agents_used = []
    final_answer = ""

    events = graph.stream(
        {"messages": [{"role": "user", "content": request.query}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            last = event["messages"][-1]
            name = getattr(last, "name", None)
            if name and name not in agents_used:
                agents_used.append(name)
            if name == "Final_Aggregator_Agent":
                final_answer = last.content

    return AnalyzeResponse(
        answer=final_answer,
        agents_used=agents_used,
        thread_id=request.thread_id,
    )


@app.post("/analyze/stream")
def analyze_stream(request: AnalyzeRequest, graph=Depends(get_graph)):
    """
    Stream the multi-agent graph execution as Server-Sent Events (SSE).

    Each SSE event has a `type` field:
    - "agent_start"  : an agent has begun processing
    - "agent_done"   : an agent has finished, includes its output
    - "final_answer" : the Final Aggregator's complete response
    - "done"         : stream is complete
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    def event_stream() -> AsyncIterator[str]:
        events = graph.stream(
            {"messages": [{"role": "user", "content": request.query}]},
            config,
            stream_mode="values",
        )
        seen_agents = set()

        for event in events:
            if "messages" not in event:
                continue

            last = event["messages"][-1]
            name = getattr(last, "name", None)

            if not name or name in seen_agents:
                continue

            seen_agents.add(name)

            if name == "Router_Agent":
                data = json.dumps({"type": "agent_start", "agent": "Router_Agent"})
                yield f"data: {data}\n\n"

            elif name == "Final_Aggregator_Agent":
                data = json.dumps({
                    "type": "final_answer",
                    "agent": "Final_Aggregator_Agent",
                    "content": last.content,
                })
                yield f"data: {data}\n\n"

            else:
                # Specialist agent completed
                data = json.dumps({
                    "type": "agent_done",
                    "agent": name,
                    "content": last.content,
                })
                yield f"data: {data}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
