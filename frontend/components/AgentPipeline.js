import styles from "./AgentPipeline.module.css";

// The full list of agents in the order they appear in the pipeline.
// This never changes — it's always the same 7 agents.
const ALL_AGENTS = [
  { id: "Router_Agent",                label: "Router Agent",              icon: "🔀" },
  { id: "Fundamental_Analysis_Agent",  label: "Fundamental Analysis",      icon: "📊" },
  { id: "Technical_Analysis_Agent",    label: "Technical Analysis",        icon: "📈" },
  { id: "Sentiment_Analysis_Agent",    label: "Sentiment Analysis",        icon: "📰" },
  { id: "Risk_Assessment_Agent",       label: "Risk Assessment",           icon: "⚠️"  },
  { id: "Real_Estate_Agent",           label: "Real Estate",               icon: "🏠" },
  { id: "Final_Aggregator_Agent",      label: "Final Aggregator",          icon: "🧠" },
];

/**
 * AgentPipeline shows the full agent architecture.
 * Each agent card changes colour based on its current status:
 *   idle   → grey  (not involved in this query)
 *   active → blue  (currently working)
 *   done   → green (finished)
 *
 * Props:
 *   agentStates: object — e.g. { Router_Agent: "active", Technical_Analysis_Agent: "done" }
 */
export default function AgentPipeline({ agentStates }) {
  return (
    <div className={styles.pipeline}>
      <h2 className={styles.title}>Agent Pipeline</h2>

      {ALL_AGENTS.map((agent, index) => {
        // Look up this agent's current status. Default to "idle" if not set.
        const status = agentStates[agent.id] || "idle";

        return (
          <div key={agent.id}>
            {/* The agent card — its CSS class changes based on status */}
            <div className={`${styles.card} ${styles[status]}`}>
              <span className={styles.icon}>{agent.icon}</span>
              <span className={styles.label}>{agent.label}</span>
              <span className={styles.statusDot} />
            </div>

            {/* Draw an arrow between cards, but not after the last one */}
            {index < ALL_AGENTS.length - 1 && (
              <div className={styles.arrow}>↓</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
