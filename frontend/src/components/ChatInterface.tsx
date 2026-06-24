"use client";

import { KeyboardEvent, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronUp, RefreshCw, Send, Zap } from "lucide-react";
import type { Citation, Chunk } from "@/lib/api";
import { useChat } from "@/hooks/useChat";
import { LoadingDots } from "./LoadingDots";

interface ChatInterfaceProps {
  selectedDocIds: string[];
  onCitations: (citations: Citation[], chunks: Chunk[]) => void;
}

export function ChatInterface({ selectedDocIds, onCitations }: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const { messages, loading, error, sendMessage } = useChat();
  const [expandedRewrites, setExpandedRewrites] = useState<Set<string>>(new Set());

  async function submit() {
    const message = await sendMessage(input, selectedDocIds.length ? selectedDocIds : null);
    if (message?.citations) {
      onCitations(message.citations, message.chunks_used || []);
    }
    setInput("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  function toggleRewrites(messageId: string) {
    setExpandedRewrites((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      return next;
    });
  }

  return (
    <section className="flex h-full flex-col">
      <div className="border-b border-line px-5 py-3">
        <h1 className="text-lg font-semibold text-ink">Ask My Docs</h1>
        <p className="text-sm text-stone-600">{selectedDocIds.length || "All"} document scope</p>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[78%] rounded-md border px-4 py-3 text-sm leading-6 ${
                message.role === "user"
                  ? "border-ink bg-ink text-white"
                  : "border-line bg-white text-ink shadow-sm"
              }`}
            >
              {/* Self-correction badge */}
              {message.role === "assistant" && (message.retry_count ?? 0) > 0 && (
                <div className="mb-3 flex items-center gap-2 rounded-md border border-blue-300 bg-blue-50 px-3 py-2 text-blue-900">
                  <RefreshCw className="h-4 w-4" aria-hidden />
                  <span>
                    Self-corrected after {message.retry_count} {message.retry_count === 1 ? "retry" : "retries"}
                  </span>
                </div>
              )}

              {/* Insufficient context warning */}
              {message.has_sufficient_context === false && (
                <div className="mb-3 flex items-center gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900">
                  <AlertTriangle className="h-4 w-4" aria-hidden />
                  <span>
                    {(message.retry_count ?? 0) >= 3
                      ? "Best-effort answer after max retries — documents may not cover this topic."
                      : "Documents don\u0027t contain enough information to answer this."}
                  </span>
                </div>
              )}

              <p className="whitespace-pre-wrap">{message.content}</p>

              {message.role === "assistant" && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onCitations(message.citations || [], message.chunks_used || [])}
                    className="rounded-md border border-line px-2 py-1 text-xs font-medium text-stone-700 hover:bg-stone-100"
                  >
                    Sources ({message.citations?.length || 0})
                  </button>
                  {typeof message.latency_ms === "number" && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-1 text-xs text-fern">
                      <Zap className="h-3 w-3" aria-hidden />
                      {(message.latency_ms / 1000).toFixed(1)}s
                    </span>
                  )}
                  {(message.retry_count ?? 0) > 0 && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2 py-1 text-xs text-blue-700">
                      <RefreshCw className="h-3 w-3" aria-hidden />
                      {message.retry_count} {message.retry_count === 1 ? "retry" : "retries"}
                    </span>
                  )}
                </div>
              )}

              {/* Rewrite history (collapsible) */}
              {message.rewrite_history && message.rewrite_history.length > 0 && (
                <div className="mt-2">
                  <button
                    type="button"
                    onClick={() => toggleRewrites(message.id)}
                    className="inline-flex items-center gap-1 text-xs font-medium text-stone-500 hover:text-stone-700"
                  >
                    {expandedRewrites.has(message.id) ? (
                      <ChevronUp className="h-3 w-3" />
                    ) : (
                      <ChevronDown className="h-3 w-3" />
                    )}
                    Query rewrites ({message.rewrite_history.length})
                  </button>
                  {expandedRewrites.has(message.id) && (
                    <div className="mt-1 space-y-1 rounded-md border border-stone-200 bg-stone-50 p-2">
                      {message.rewrite_history.map((rewrite, idx) => (
                        <div key={idx} className="text-xs text-stone-600">
                          <span className="font-mono text-stone-400">#{idx + 1}</span>{" "}
                          <span className="italic">&ldquo;{rewrite}&rdquo;</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="inline-flex items-center gap-2 rounded-md border border-line bg-white px-3 py-2 text-sm text-stone-700">
            Searching documents <LoadingDots />
          </div>
        )}

        {error && <div className="rounded-md border border-clay bg-red-50 px-3 py-2 text-sm text-clay">{error}</div>}
      </div>

      <div className="border-t border-line bg-paper px-5 py-4">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your question..."
            className="min-h-11 flex-1 resize-none rounded-md border border-line bg-white px-3 py-2 text-sm outline-none focus:border-fern"
          />
          <button
            type="button"
            onClick={() => void submit()}
            disabled={loading || !input.trim()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-md bg-fern text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"
            title="Send question"
            aria-label="Send question"
          >
            <Send className="h-5 w-5" aria-hidden />
          </button>
        </div>
      </div>
    </section>
  );
}
