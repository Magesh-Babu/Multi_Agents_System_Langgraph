import { useEffect, useRef } from "react";
import styles from "./ChatPanel.module.css";

/**
 * ChatPanel shows the full conversation history.
 *
 * Props:
 *   messages:  array of { role: "user"|"assistant", content: string }
 *   isLoading: boolean — true while agents are still working
 */
export default function ChatPanel({ messages, isLoading }) {
  // useRef gives us a direct reference to a DOM element.
  // We use it to automatically scroll to the bottom when new messages arrive.
  const bottomRef = useRef(null);

  // useEffect runs after React updates the screen.
  // Here, every time `messages` changes, we scroll to the bottom.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className={styles.panel}>
      {/* Show a welcome message when the chat is empty */}
      {messages.length === 0 && !isLoading && (
        <div className={styles.empty}>
          <p>👋 Ask me about stocks, technical indicators, market sentiment, or Swedish housing prices.</p>
        </div>
      )}

      {/* Render each message in the conversation */}
      {messages.map((msg, index) => (
        <div
          key={index}
          // User messages align right, assistant messages align left
          className={`${styles.message} ${styles[msg.role]}`}
        >
          <div className={styles.bubble}>
            {/* Show a label above each bubble */}
            <span className={styles.roleLabel}>
              {msg.role === "user" ? "You" : "Assistant"}
            </span>
            {/* The actual message text */}
            <p>{msg.content}</p>
          </div>
        </div>
      ))}

      {/* Show a "thinking" indicator while agents are running */}
      {isLoading && (
        <div className={`${styles.message} ${styles.assistant}`}>
          <div className={styles.bubble}>
            <span className={styles.roleLabel}>Assistant</span>
            <div className={styles.thinking}>
              <span />
              <span />
              <span />
            </div>
          </div>
        </div>
      )}

      {/* Invisible element at the bottom — we scroll to this */}
      <div ref={bottomRef} />
    </div>
  );
}
