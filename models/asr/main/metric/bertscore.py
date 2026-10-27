import torch
import evaluate
import datasets
from bert_score import score


_DESCRIPTION = "BERTScore measures the semantic similarity between machine-generated text and reference translations."

_KWARGS_DESCRIPTION = """
Compute BERTScore between predictions and references.

Args:
    predictions (list of str): Predicted sentences.
    references (list of str): Reference sentences.
    model_type (str, optional): Pretrained model used to compute BERTScore embeddings.
        Defaults to 'microsoft/deberta-xlarge-mnli'.

Returns:
    dict: {'bertscore_precision': float, 'bertscore_recall': float, 'bertscore_f1': float}
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BERTScore(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=(
                "Zhang et al. (2020). "
                "BERTScore: Evaluating Text Generation with BERT. "
                "https://arxiv.org/abs/1904.09675"
            ),
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("string"),
                "references": datasets.Value("string"),
            }),
        )

    def _compute(
        self,
        predictions=None,
        references=None,
        model_type="microsoft/deberta-xlarge-mnli",
        lang="en",
        batch_size=32,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if predictions is None or references is None:
            raise ValueError("Predictions and references must be provided.")
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        P, R, F1 = score(
            predictions,
            references,
            lang=lang,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=False,
        )

        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
