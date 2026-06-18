#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../backend"
python - <<'PY'
import asyncio
import json
import sys

from evaluation.dataset_generator import load_golden_dataset
from evaluation.eval_runner import run_evaluation
from generation.chain import run_rag_chain

dataset = load_golden_dataset()
if dataset["num_questions"] == 0:
    print("No golden dataset available — skipping eval")
    sys.exit(0)

async def main():
    results = await run_evaluation(run_rag_chain)
    with open("eval_results.json", "w") as f:
        json.dump(results, f)
    print("Eval results:", results)
    if not results["passed"]:
        sys.exit(1)

asyncio.run(main())
PY
