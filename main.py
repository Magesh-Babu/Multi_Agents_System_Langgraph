from dotenv import load_dotenv
load_dotenv()

from stock_analyst.graph.builder import compile_graph


def main():
    graph = compile_graph()
    thread_id = 1

    print("Stock Analyst — Multi-Agent System")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        config = {"configurable": {"thread_id": str(thread_id)}}
        events = graph.stream(
            {"messages": [{"role": "user", "content": query}]},
            config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                last = event["messages"][-1]
                name = getattr(last, "name", None)
                if name and name != "Router_Agent":
                    print(f"  [{name}] working...")
                if name == "Final_Aggregator_Agent":
                    print(f"\nAssistant: {last.content}\n")

        thread_id += 1


if __name__ == "__main__":
    main()
