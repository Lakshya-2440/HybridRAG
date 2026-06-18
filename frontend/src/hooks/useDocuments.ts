"use client";

import { useCallback, useEffect, useState } from "react";
import {
  deleteDocument,
  getDocument,
  ingestUrl,
  listDocuments,
  type DocumentRecord,
  uploadDocument,
} from "@/lib/api";

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [datasetBanner, setDatasetBanner] = useState(false);

  const refresh = useCallback(async () => {
    const docs = await listDocuments();
    setDocuments(docs);
    setError(null);
    setLoading(false);
    return docs;
  }, []);

  useEffect(() => {
    refresh().catch((err) => {
      setError(err.message);
      setLoading(false);
    });
  }, [refresh]);

  const pollDocument = useCallback(
    (docId: string) => {
      const interval = window.setInterval(async () => {
        try {
          const doc = await getDocument(docId);
          setDocuments((current) => current.map((item) => (item.id === docId ? doc : item)));
          if (doc.status === "ready" || doc.status === "failed") {
            window.clearInterval(interval);
            setUploadProgress((current) => ({ ...current, [docId]: 100 }));
            if (doc.status === "ready") {
              setDatasetBanner(true);
            }
            await refresh();
          }
        } catch (err) {
          window.clearInterval(interval);
          setError(err instanceof Error ? err.message : "Polling failed");
        }
      }, 2000);
    },
    [refresh],
  );

  const uploadFiles = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        try {
          const response = await uploadDocument(file);
          setUploadProgress((current) => ({ ...current, [response.doc_id]: 60 }));
          await refresh();
          pollDocument(response.doc_id);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Upload failed");
        }
      }
    },
    [pollDocument, refresh],
  );

  const submitUrl = useCallback(
    async (url: string) => {
      const response = await ingestUrl(url);
      setUploadProgress((current) => ({ ...current, [response.doc_id]: 60 }));
      await refresh();
      pollDocument(response.doc_id);
    },
    [pollDocument, refresh],
  );

  const removeDocument = useCallback(
    async (docId: string) => {
      await deleteDocument(docId);
      await refresh();
    },
    [refresh],
  );

  return {
    documents,
    loading,
    error,
    uploadProgress,
    datasetBanner,
    refresh,
    uploadFiles,
    submitUrl,
    removeDocument,
  };
}
