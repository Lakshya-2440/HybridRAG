"use client";

import { CheckCircle2, Clock, FileText, Trash2, XCircle } from "lucide-react";
import type { DocumentRecord } from "@/lib/api";

interface DocumentListProps {
  documents: DocumentRecord[];
  selectedIds: string[];
  onToggle: (docId: string) => void;
  onDelete: (docId: string) => void;
}

function StatusBadge({ status }: { status: string }) {
  if (status === "ready") {
    return <CheckCircle2 className="h-4 w-4 text-fern" aria-label="ready" />;
  }
  if (status === "failed") {
    return <XCircle className="h-4 w-4 text-clay" aria-label="failed" />;
  }
  return <Clock className="h-4 w-4 text-honey" aria-label="processing" />;
}

export function DocumentList({ documents, selectedIds, onToggle, onDelete }: DocumentListProps) {
  return (
    <div className="space-y-2">
      {documents.map((doc) => {
        const selected = selectedIds.includes(doc.id);
        return (
          <div
            key={doc.id}
            className={`rounded-md border p-2 text-sm ${
              selected ? "border-fern bg-emerald-50" : "border-line bg-white"
            }`}
          >
            <label className="flex cursor-pointer items-start gap-2">
              <input
                type="checkbox"
                className="mt-1 h-4 w-4 accent-fern"
                checked={selected}
                onChange={() => onToggle(doc.id)}
              />
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-2">
                  <FileText className="h-4 w-4 shrink-0 text-stone-500" aria-hidden />
                  <span className="truncate font-medium text-ink">{doc.filename}</span>
                  <StatusBadge status={doc.status} />
                </span>
                <span className="mt-1 block text-xs text-stone-600">
                  {doc.file_type.toUpperCase()} - {doc.chunk_count} chunks
                </span>
                {doc.error_message && <span className="mt-1 block text-xs text-clay">{doc.error_message}</span>}
              </span>
            </label>
            <button
              type="button"
              className="mt-2 inline-flex h-8 w-8 items-center justify-center rounded-md border border-line text-stone-600 hover:bg-stone-100"
              onClick={() => onDelete(doc.id)}
              title="Delete document"
              aria-label={`Delete ${doc.filename}`}
            >
              <Trash2 className="h-4 w-4" aria-hidden />
            </button>
          </div>
        );
      })}
      {documents.length === 0 && (
        <div className="rounded-md border border-dashed border-line p-3 text-sm text-stone-600">
          No documents uploaded.
        </div>
      )}
    </div>
  );
}
