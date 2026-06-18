"use client";

import { useCallback, useState } from "react";
import { askQuestion, type Citation, type Chunk } from "@/lib/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  chunks_used?: Chunk[];
  has_sufficient_context?: boolean;
  latency_ms?: number;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (query: string, docIds: string[] | null) => {
    const trimmed = query.trim();
    if (!trimmed || loading) return null;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    setMessages((current) => [...current, userMessage]);
    setLoading(true);
    setError(null);

    try {
      const response = await askQuestion(trimmed, docIds);
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        chunks_used: response.chunks_used,
        has_sufficient_context: response.has_sufficient_context,
        latency_ms: response.latency_ms,
      };
      setMessages((current) => [...current, assistantMessage]);
      return assistantMessage;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
      return null;
    } finally {
      setLoading(false);
    }
  }, [loading]);

  return { messages, loading, error, sendMessage };
}
