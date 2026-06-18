try:
    from ragas.metrics import answer_relevancy, faithfulness
except ImportError:  # pragma: no cover
    faithfulness = None
    answer_relevancy = None


METRIC_NAMES = ["faithfulness", "answer_relevancy"]
