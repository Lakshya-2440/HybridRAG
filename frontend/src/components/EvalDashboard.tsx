"use client";

import { useEffect, useState } from "react";
import { RefreshCcw, RotateCw } from "lucide-react";
import { getEvalResults, regenerateDataset, runEvaluation, type EvalRun } from "@/lib/api";
import { LoadingDots } from "./LoadingDots";

function ScoreBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 overflow-hidden rounded-full bg-stone-200">
        <div className="h-full bg-fern" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
      </div>
      <span className="font-mono text-xs">{value.toFixed(2)}</span>
    </div>
  );
}

export function EvalDashboard() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setRuns(await getEvalResults());
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  async function startEval() {
    setBusy(true);
    setError(null);
    try {
      await runEvaluation();
      const interval = window.setInterval(async () => {
        const next = await getEvalResults();
        setRuns(next);
        if (next.length) {
          window.clearInterval(interval);
          setBusy(false);
        }
      }, 3000);
    } catch (err) {
      setBusy(false);
      setError(err instanceof Error ? err.message : "Evaluation failed");
    }
  }

  async function regenerate() {
    setBusy(true);
    try {
      await regenerateDataset();
      window.setTimeout(() => setBusy(false), 3000);
    } catch (err) {
      setBusy(false);
      setError(err instanceof Error ? err.message : "Dataset generation failed");
    }
  }

  return (
    <main className="min-h-screen bg-paper p-6 text-ink">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pb-4">
          <div>
            <h1 className="text-xl font-semibold">Evaluation</h1>
            <p className="text-sm text-stone-600">Last 20 RAGAS runs</p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={startEval}
              disabled={busy}
              className="inline-flex h-9 items-center gap-2 rounded-md bg-fern px-3 text-sm font-medium text-white disabled:opacity-50"
            >
              <RefreshCcw className="h-4 w-4" aria-hidden />
              Run Evaluation
            </button>
            <button
              type="button"
              onClick={regenerate}
              disabled={busy}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-line bg-white px-3 text-sm font-medium text-ink disabled:opacity-50"
            >
              <RotateCw className="h-4 w-4" aria-hidden />
              Regenerate Dataset
            </button>
          </div>
        </div>

        {busy && <div className="mt-4 text-sm text-stone-700">Working <LoadingDots /></div>}
        {error && <div className="mt-4 rounded-md border border-clay bg-red-50 p-3 text-sm text-clay">{error}</div>}

        <div className="mt-5 overflow-hidden rounded-md border border-line bg-white">
          <table className="w-full border-collapse text-sm">
            <thead className="bg-stone-100 text-left text-xs uppercase tracking-wide text-stone-600">
              <tr>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Faithfulness</th>
                <th className="px-4 py-3">Answer Relevancy</th>
                <th className="px-4 py-3">Questions</th>
                <th className="px-4 py-3">Result</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id} className={run.passed ? "bg-emerald-50/50" : "bg-red-50/50"}>
                  <td className="px-4 py-3 font-mono text-xs">{new Date(run.run_timestamp).toLocaleString()}</td>
                  <td className="px-4 py-3"><ScoreBar value={run.faithfulness_score} /></td>
                  <td className="px-4 py-3"><ScoreBar value={run.answer_relevancy_score} /></td>
                  <td className="px-4 py-3">{run.num_questions}</td>
                  <td className="px-4 py-3 font-semibold">{run.passed ? "PASS" : "FAIL"}</td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-stone-600">
                    No evaluation runs yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}
