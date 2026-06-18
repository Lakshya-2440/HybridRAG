"use client";

import { ChangeEvent, DragEvent, useState } from "react";
import { LinkIcon, UploadCloud } from "lucide-react";
import { LoadingDots } from "./LoadingDots";

interface DocumentUploadProps {
  uploadProgress: Record<string, number>;
  datasetBanner: boolean;
  onFiles: (files: File[]) => Promise<void>;
  onUrl: (url: string) => Promise<void>;
}

export function DocumentUpload({ uploadProgress, datasetBanner, onFiles, onUrl }: DocumentUploadProps) {
  const [dragging, setDragging] = useState(false);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleFiles(files: FileList | File[]) {
    const accepted = Array.from(files).filter((file) => /\.(pdf|txt|md|html?)$/i.test(file.name));
    if (!accepted.length) return;
    setBusy(true);
    await onFiles(accepted);
    setBusy(false);
  }

  function onInput(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      void handleFiles(event.target.files);
    }
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    void handleFiles(event.dataTransfer.files);
  }

  async function submitUrl() {
    if (!url.trim()) return;
    setBusy(true);
    await onUrl(url.trim());
    setUrl("");
    setBusy(false);
  }

  const progressValues = Object.entries(uploadProgress).slice(-3);

  return (
    <div className="space-y-3">
      <label
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex h-28 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed px-3 text-center ${
          dragging ? "border-fern bg-emerald-50" : "border-line bg-white"
        }`}
      >
        <UploadCloud className="h-6 w-6 text-fern" aria-hidden />
        <span className="mt-2 text-sm font-medium text-ink">Upload PDF, TXT, MD, HTML</span>
        <span className="mt-1 text-xs text-stone-600">{busy ? <LoadingDots /> : "Drop files or browse"}</span>
        <input type="file" className="hidden" accept=".pdf,.txt,.md,.html,.htm" multiple onChange={onInput} />
      </label>

      <div className="flex gap-2">
        <div className="relative flex-1">
          <LinkIcon className="pointer-events-none absolute left-2 top-2.5 h-4 w-4 text-stone-500" aria-hidden />
          <input
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://docs.example.com/page"
            className="h-9 w-full rounded-md border border-line bg-white pl-8 pr-2 text-sm outline-none focus:border-fern"
          />
        </div>
        <button
          type="button"
          onClick={submitUrl}
          className="h-9 rounded-md bg-ink px-3 text-sm font-medium text-white hover:bg-black"
        >
          Ingest URL
        </button>
      </div>

      {progressValues.map(([docId, value]) => (
        <div key={docId} className="h-2 overflow-hidden rounded-full bg-stone-200" aria-label="Upload progress">
          <div className="h-full bg-fern transition-all" style={{ width: `${value}%` }} />
        </div>
      ))}

      {datasetBanner && (
        <div className="rounded-md border border-amber-300 bg-amber-50 p-2 text-xs text-amber-900">
          Generating evaluation dataset in background...
        </div>
      )}
    </div>
  );
}
