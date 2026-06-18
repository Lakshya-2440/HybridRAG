import { AlertTriangle, CheckCircle2, FileText } from "lucide-react";
import type { Citation, Chunk } from "@/lib/api";

interface CitationCardProps {
  citation: Citation;
  chunk?: Chunk;
}

export function CitationCard({ citation, chunk }: CitationCardProps) {
  const score = chunk?.rerank_score ?? chunk?.score ?? null;

  return (
    <article
      className={`rounded-md border bg-white p-3 shadow-sm ${
        citation.verified ? "border-line" : "border-orange-400"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <FileText className="h-4 w-4 shrink-0 text-clay" aria-hidden />
          <p className="truncate text-sm font-semibold text-ink">
            {citation.source} - Page {citation.page}, Chunk {citation.chunk_index}
          </p>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2 text-xs">
        {citation.verified ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-1 text-fern">
            <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />
            Verified
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 rounded-full bg-orange-50 px-2 py-1 text-orange-700">
            <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
            Unverified
          </span>
        )}
        {score !== null && <span className="font-mono text-stone-600">Score: {score.toFixed(2)}</span>}
      </div>
      <p className="mt-3 line-clamp-5 text-sm leading-6 text-stone-700">
        &quot;{citation.excerpt || chunk?.text?.slice(0, 200) || "No excerpt available."}&quot;
      </p>
    </article>
  );
}
