import evaluate
import datasets


_DESCRIPTION = "Exact Match Rate (EMR) measures the proportion of predictions that exactly match the reference."

_KWARGS_DESCRIPTION = """
Compute Exact Match Rate of transcribed segments against references.

Args:
    references: List of reference texts.
    predictions: List of predicted texts.

Returns:
    float: Proportion of exactly matched predictions.
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EMR(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("string"),
                "references": datasets.Value("string"),
            }),
        )

    def _compute(self, predictions=None, references=None):
        if predictions is None or references is None:
            raise ValueError("Predictions and references must be provided.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        
        correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        return correct / len(predictions)
