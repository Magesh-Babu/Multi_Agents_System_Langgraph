import { useState, useCallback } from "react";
import Head from "next/head";
import AgentPipeline from "../components/AgentPipeline";
import ChatPanel from "../components/ChatPanel";
import { streamAnalysis } from "../lib/stream";
import styles from "../styles/Home.module.css";

// Initial state for the agent pipeline — all agents start as idle.
// We reset to this before every new query.
const INITIAL_AGENT_STATES = {
  Router_Agent:               "idle",
  Fundamental_Analysis_Agent: "idle",
  Technical_Analysis_Agent:   "idle",
  Sentiment_Analysis_Agent:   "idle",
  Risk_Assessment_Agent:      "idle",
  Real_Estate_Agent:          "idle",
  Final_Aggregator_Agent:     "idle",
};

export default function Home() {
  // ── State ──────────────────────────────────────────────────────────────────
  // useState(initialValue) creates a piece of state.
  // React re-renders the page whenever any of these change.

  // The text the user is currently typing in the input box.
  const [query, setQuery] = useState("");

  // Full conversation history — array of { role, content } objects.
  const [messages, setMessages] = useState([]);

  // Current status of each agent in the pipeline.
  const [agentStates, setAgentStates] = useState(INITIAL_AGENT_STATES);

  // True while we are waiting for the agents to finish.
  const [isLoading, setIsLoading] = useState(false);

  // A counter we increment per query to generate unique thread IDs.
  const [threadCounter, setThreadCounter] = useState(1);

  // ── Submit handler ─────────────────────────────────────────────────────────
  // useCallback memoises the function so it doesn't get recreated on every render.
  const handleSubmit = useCallback(async () => {
    const trimmed = query.trim();
    if (!trimmed || isLoading) return;

    // 1. Add the user's message to the chat history.
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);

    // 2. Clear the input box.
    setQuery("");

    // 3. Reset the agent pipeline — all idle for the new query.
    setAgentStates(INITIAL_AGENT_STATES);

    // 4. Show the loading indicator in the chat.
    setIsLoading(true);

    // 5. Create a unique thread ID for this conversation turn.
    const threadId = `thread-${threadCounter}`;
    setThreadCounter((n) => n + 1);

    try {
      // 6. Connect to the FastAPI SSE stream.
      //    onEvent is called every time an agent event arrives.
      await streamAnalysis(trimmed, threadId, (event) => {
        if (event.type === "agent_start") {
          // Router has started — mark it active.
          setAgentStates((prev) => ({
            ...prev,
            [event.agent]: "active",
          }));
        }

        if (event.type === "agent_done") {
          // A specialist agent finished — mark Router done, this agent done.
          setAgentStates((prev) => ({
            ...prev,
            Router_Agent: "done",
            [event.agent]: "done",
          }));
        }

        if (event.type === "final_answer") {
          // Final Aggregator finished — mark it done and add answer to chat.
          setAgentStates((prev) => ({
            ...prev,
            Final_Aggregator_Agent: "done",
          }));
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: event.content },
          ]);
        }

        if (event.type === "done") {
          // Stream is fully closed — hide the loading indicator.
          setIsLoading(false);
        }
      });
    } catch (err) {
      // If the API is unreachable, show an error message in the chat.
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Could not reach the API. Make sure the FastAPI server is running on localhost:8000." },
      ]);
      setIsLoading(false);
    }
  }, [query, isLoading, threadCounter]);

  // Allow submitting with the Enter key (Shift+Enter adds a newline).
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>Stock Analyst — Multi-Agent System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className={styles.page}>

        {/* Header bar across the top */}
        <header className={styles.header}>
          <h1>Stock Analyst</h1>
          <span className={styles.subtitle}>Multi-Agent System · LangGraph + FastAPI</span>
        </header>

        {/* Main content area: left panel + right panel */}
        <main className={styles.main}>

          {/* LEFT: Agent pipeline — shows which agents are active */}
          <aside className={styles.sidebar}>
            <AgentPipeline agentStates={agentStates} />
          </aside>

          {/* RIGHT: Chat history + input */}
          <section className={styles.chat}>
            <ChatPanel messages={messages} isLoading={isLoading} />

            {/* Input area at the bottom of the chat */}
            <div className={styles.inputArea}>
              <textarea
                className={styles.input}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about a stock, technical indicators, market sentiment, or Swedish housing prices..."
                rows={2}
                disabled={isLoading}
              />
              <button
                className={styles.sendBtn}
                onClick={handleSubmit}
                disabled={isLoading || !query.trim()}
              >
                {isLoading ? "..." : "Send"}
              </button>
            </div>
          </section>

        </main>
      </div>
    </>
  );
}
