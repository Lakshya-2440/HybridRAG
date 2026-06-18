"use client";

import { useState } from "react";
import { CitationCard } from "@/components/CitationCard";
import { ChatInterface } from "@/components/ChatInterface";
import { DocumentList } from "@/components/DocumentList";
import { DocumentUpload } from "@/components/DocumentUpload";
import { useDocuments } from "@/hooks/useDocuments";
import type { Citation, Chunk } from "@/lib/api";

export default function HomePage() {
  const {
    documents,
    loading,
    error,
    uploadProgress,
    datasetBanner,
    uploadFiles,
    submitUrl,
    removeDocument,
  } = useDocuments();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);

  function toggleDoc(docId: string) {
    setSelectedIds((current) =>
      current.includes(docId) ? current.filter((id) => id !== docId) : [...current, docId],
    );
  }

  function showCitations(nextCitations: Citation[], nextChunks: Chunk[]) {
    setCitations(nextCitations);
    setChunks(nextChunks);
  }

  return (
    <main className="h-screen overflow-hidden bg-paper text-ink">
      <div className="grid h-full grid-cols-1 md:grid-cols-[250px_minmax(0,1fr)_300px]">
        <aside className="flex min-h-0 flex-col border-r border-line bg-paper">
          <div className="border-b border-line p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-600">Documents</h2>
          </div>
          <div className="space-y-4 overflow-y-auto p-4">
            <DocumentUpload
              uploadProgress={uploadProgress}
              datasetBanner={datasetBanner}
              onFiles={uploadFiles}
              onUrl={submitUrl}
            />
            {loading && <p className="text-sm text-stone-600">Loading documents...</p>}
            {error && <p className="rounded-md border border-clay bg-red-50 p-2 text-sm text-clay">{error}</p>}
            <DocumentList
              documents={documents}
              selectedIds={selectedIds}
              onToggle={toggleDoc}
              onDelete={removeDocument}
            />
          </div>
        </aside>

        <ChatInterface selectedDocIds={selectedIds} onCitations={showCitations} />

        <aside className="hidden min-h-0 flex-col border-l border-line bg-[#fbfaf7] md:flex">
          <div className="border-b border-line p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-600">Citations</h2>
          </div>
          <div className="flex-1 space-y-3 overflow-y-auto p-4">
            {citations.map((citation, index) => (
              <CitationCard
                key={`${citation.source}-${citation.page}-${citation.chunk_index}-${index}`}
                citation={citation}
                chunk={chunks[index]}
              />
            ))}
            {citations.length === 0 && (
              <div className="rounded-md border border-dashed border-line p-3 text-sm text-stone-600">
                Source cards appear after an answer.
              </div>
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}
