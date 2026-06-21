const API_BASE = "/api";

export interface DocumentRecord {
  id: string;
  filename: string;
  file_type: string;
  status: "processing" | "ready" | "failed" | string;
  chunk_count: number;
  upload_timestamp: string;
  error_message?: string | null;
}

export interface Citation {
  source: string;
  page: number;
  chunk_index: number;
  verified: boolean;
  excerpt: string;
}

export interface Chunk {
  text: string;
  metadata: Record<string, unknown>;
  score?: number | null;
  bm25_score?: number | null;
  rerank_score?: number | null;
}

export interface QueryResponse {
  answer: string;
  has_sufficient_context: boolean;
  citations: Citation[];
  chunks_used: Chunk[];
  latency_ms: number;
}

export interface EvalRun {
  id: string;
  run_timestamp: string;
  faithfulness_score: number;
  answer_relevancy_score: number;
  num_questions: number;
  passed: boolean;
  raw_results: string;
  triggered_by: string;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options?.headers || {}),
    },
  });
  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const text = await response.text();
      try {
        const error = JSON.parse(text);
        message = error.detail || error.message || message;
      } catch {
        message = text || message;
      }
    } catch {
      // Ignore text reading errors
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function uploadDocument(file: File): Promise<{ doc_id: string; status: string }> {
  const body = new FormData();
  body.append("file", file);
  return request("/documents/upload", { method: "POST", body });
}

export async function ingestUrl(url: string): Promise<{ doc_id: string; status: string }> {
  return request("/documents/upload", { method: "POST", body: JSON.stringify({ url }) });
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  return request("/documents");
}

export async function getDocument(docId: string): Promise<DocumentRecord> {
  return request(`/documents/${docId}`);
}

export async function deleteDocument(docId: string): Promise<{ deleted: boolean }> {
  return request(`/documents/${docId}`, { method: "DELETE" });
}

export async function askQuestion(query: string, docIds: string[] | null): Promise<QueryResponse> {
  return request("/query", {
    method: "POST",
    body: JSON.stringify({ query, doc_ids: docIds }),
  });
}

export async function runEvaluation(): Promise<{ eval_id: string; status: string }> {
  return request("/eval/run", {
    method: "POST",
    body: JSON.stringify({ triggered_by: "manual" }),
  });
}

export async function getEvalResults(): Promise<EvalRun[]> {
  return request("/eval/results");
}

export async function regenerateDataset(): Promise<{ status: string }> {
  return request("/eval/generate", { method: "POST" });
}
