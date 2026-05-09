const API_URL = "http://localhost:8000";

/**
 * Sends a query to the FastAPI streaming endpoint and calls
 * onEvent() for every agent event that comes back.
 *
 * @param {string} query      - The user's question
 * @param {string} threadId   - Unique ID for this conversation thread
 * @param {function} onEvent  - Called with each parsed event object
 */
export async function streamAnalysis(query, threadId, onEvent) {
  // Step 1: Send a POST request to the FastAPI /analyze/stream endpoint.
  // The response body is a continuous stream — it doesn't close immediately.
  const response = await fetch(`${API_URL}/analyze/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, thread_id: threadId }),
  });

  // Step 2: Get a "reader" — this lets us read the stream piece by piece
  // instead of waiting for the entire response to finish.
  const reader = response.body.getReader();

  // Step 3: TextDecoder converts raw bytes from the stream into readable text.
  const decoder = new TextDecoder();

  // Step 4: Keep reading chunks of data until the stream closes.
  while (true) {
    const { done, value } = await reader.read();

    // When done is true, the server closed the stream — we stop.
    if (done) break;

    // Decode the raw bytes into a string (may contain multiple lines).
    const text = decoder.decode(value);

    // Step 5: SSE format sends lines like: "data: {...json...}\n\n"
    // We split by newline and process each line separately.
    const lines = text.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          // Strip the "data: " prefix and parse the JSON.
          const event = JSON.parse(line.slice(6));
          // Call the handler with the parsed event object.
          onEvent(event);
        } catch {
          // Ignore malformed lines (can happen at chunk boundaries).
        }
      }
    }
  }
}
