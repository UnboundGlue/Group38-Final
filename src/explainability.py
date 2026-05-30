"""Explainability module: SHAP and LIME explanations for authorship attribution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .models import ErrorAnalysisReport
from .tokeniser import SubwordTokeniser


# ---------------------------------------------------------------------------
# Explanation dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ShapExplanation:
    """SHAP token-level attributions for a single text."""
    text: str
    token_ids: list[int]
    shap_values: np.ndarray   # shape [num_tokens] — attribution per token position
    predicted_class: int


@dataclass
class LimeExplanation:
    """LIME local surrogate explanation for a single text."""
    text: str
    explanation: dict[str, float]   # token → weight
    predicted_class: int


# ---------------------------------------------------------------------------
# ExplainabilityModule
# ---------------------------------------------------------------------------

class ExplainabilityModule:
    """Generate SHAP/LIME explanations and perform error analysis."""

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def explain_shap(
        self,
        model: torch.nn.Module,
        tokeniser: SubwordTokeniser,
        texts: list[str],
        background_texts: list[str],
        max_length: int = 256,
    ) -> list[ShapExplanation]:
        """Compute token-level SHAP attributions for each text.

        Uses ``shap.KernelExplainer`` with a wrapper that converts token ID
        arrays to softmax probabilities.

        Args:
            model: Trained CNNLSTMModel (or any ``nn.Module`` with the same
                   forward signature).
            tokeniser: Trained SubwordTokeniser.
            texts: Input texts to explain.
            background_texts: Background corpus used to build the SHAP baseline.
            max_length: Sequence length used for encoding.

        Returns:
            One :class:`ShapExplanation` per input text.
        """
        import shap  # lazy import — optional dependency

        model.eval()

        # Encode background and input texts
        background_ids = tokeniser.batch_encode(background_texts, max_length=max_length)
        input_ids = tokeniser.batch_encode(texts, max_length=max_length)

        # Wrapper: float array [N, max_length] → softmax probabilities [N, C]
        def _predict(token_array: np.ndarray) -> np.ndarray:
            token_tensor = torch.tensor(token_array, dtype=torch.long)
            with torch.no_grad():
                logits = model(token_tensor)
                probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()

        explainer = shap.KernelExplainer(_predict, background_ids.astype(float))
        # shap_values is a list of arrays (one per class) each of shape [N, max_length]
        shap_values_all = explainer.shap_values(input_ids.astype(float), nsamples=100)

        results: list[ShapExplanation] = []
        for i, text in enumerate(texts):
            token_ids = input_ids[i].tolist()

            # Determine predicted class from model
            token_tensor = torch.tensor(input_ids[i : i + 1], dtype=torch.long)
            with torch.no_grad():
                logits = model(token_tensor)
            predicted_class = int(logits.argmax(dim=-1).item())

            # shap_values_all: list[ndarray shape (N, max_length)] indexed by class
            # Take attributions for the predicted class
            if isinstance(shap_values_all, list):
                sv = shap_values_all[predicted_class][i]  # shape [max_length]
            else:
                # Single-output case (binary): shape [N, max_length]
                sv = shap_values_all[i]

            results.append(ShapExplanation(
                text=text,
                token_ids=token_ids,
                shap_values=np.array(sv, dtype=float),
                predicted_class=predicted_class,
            ))

        return results

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------

    def explain_lime(
        self,
        model: torch.nn.Module,
        tokeniser: SubwordTokeniser,
        text: str,
        num_samples: int = 500,
        max_length: int = 256,
    ) -> LimeExplanation:
        """Compute a LIME local surrogate explanation for a single text.

        Args:
            model: Trained CNNLSTMModel.
            tokeniser: Trained SubwordTokeniser.
            text: The text to explain.
            num_samples: Number of perturbed samples for LIME.
            max_length: Sequence length used for encoding.

        Returns:
            A :class:`LimeExplanation` with token → weight mapping.
        """
        from lime.lime_text import LimeTextExplainer  # lazy import

        model.eval()

        # Determine number of classes from a single forward pass
        token_ids = tokeniser.encode(text, max_length=max_length)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(token_tensor)
        num_classes = logits.shape[-1]
        predicted_class = int(logits.argmax(dim=-1).item())

        def _predict_fn(texts_batch: list[str]) -> np.ndarray:
            """Tokenise a batch of (perturbed) texts and return probabilities."""
            encoded = tokeniser.batch_encode(texts_batch, max_length=max_length)
            tensor = torch.tensor(encoded, dtype=torch.long)
            with torch.no_grad():
                logits_batch = model(tensor)
                probs = torch.softmax(logits_batch, dim=-1)
            return probs.cpu().numpy()

        lime_explainer = LimeTextExplainer(class_names=list(range(num_classes)))
        lime_result = lime_explainer.explain_instance(
            text,
            _predict_fn,
            num_features=20,
            num_samples=num_samples,
            labels=[predicted_class],
        )

        # Build token → weight dict for the predicted class
        explanation_dict: dict[str, float] = dict(
            lime_result.as_list(label=predicted_class)
        )

        return LimeExplanation(
            text=text,
            explanation=explanation_dict,
            predicted_class=predicted_class,
        )

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------

    def error_analysis(
        self,
        explanations: list[ShapExplanation | LimeExplanation],
        predictions: list[int],
        labels: list[int],
        top_k: int = 5,
    ) -> ErrorAnalysisReport:
        """Aggregate misclassification patterns from explanations.

        Args:
            explanations: One explanation per sample (SHAP or LIME).
            predictions: Predicted class indices, length N.
            labels: Ground-truth class indices, length N.
            top_k: Number of top influential tokens to surface per author pair.

        Returns:
            :class:`ErrorAnalysisReport` with misclassified indices, top tokens
            per confusion pair, and confusion pairs ranked by rate.

        Raises:
            ValueError: If there are no misclassified samples.
        """
        if len(explanations) != len(predictions) or len(predictions) != len(labels):
            raise ValueError(
                "explanations, predictions, and labels must all have the same length."
            )

        # Identify misclassified indices
        misclassified_indices = [
            i for i, (p, l) in enumerate(zip(predictions, labels)) if p != l
        ]

        if not misclassified_indices:
            raise ValueError(
                "error_analysis() requires at least one misclassified sample, "
                "but all predictions match the ground-truth labels."
            )

        # Aggregate token attributions per (true_label, predicted) pair
        # token_scores: (true, pred) → {token: [scores]}
        token_scores: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        for i in misclassified_indices:
            true_label = labels[i]
            pred_label = predictions[i]
            pair = (true_label, pred_label)
            pair_counts[pair] += 1

            exp = explanations[i]

            if isinstance(exp, ShapExplanation):
                # Map token positions to decoded tokens
                token_ids = exp.token_ids
                shap_vals = exp.shap_values
                # Group by unique token id, accumulate absolute attribution
                for pos, (tid, sv) in enumerate(zip(token_ids, shap_vals)):
                    if tid == 0:  # skip PAD
                        continue
                    token_str = str(tid)  # use id as key; decode if tokeniser available
                    token_scores[pair][token_str].append(abs(float(sv)))

            elif isinstance(exp, LimeExplanation):
                for token_str, weight in exp.explanation.items():
                    token_scores[pair][token_str].append(abs(float(weight)))

        # Build top-k tokens per pair
        top_tokens_per_pair: dict[tuple[int, int], list[str]] = {}
        for pair, token_dict in token_scores.items():
            # Average attribution per token, then take top-k
            avg_scores = {tok: float(np.mean(scores)) for tok, scores in token_dict.items()}
            sorted_tokens = sorted(avg_scores, key=avg_scores.get, reverse=True)  # type: ignore[arg-type]
            top_tokens_per_pair[pair] = sorted_tokens[:top_k]

        # Compute confusion rates: count(pair) / total_samples_with_true_label
        true_label_counts: dict[int, int] = defaultdict(int)
        for l in labels:
            true_label_counts[l] += 1

        confusion_pairs: list[tuple[int, int, float]] = []
        for pair, count in pair_counts.items():
            true_label = pair[0]
            total = true_label_counts[true_label]
            rate = count / total if total > 0 else 0.0
            confusion_pairs.append((pair[0], pair[1], rate))

        # Sort by confusion rate descending
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)

        return ErrorAnalysisReport(
            misclassified_indices=misclassified_indices,
            top_tokens_per_pair=top_tokens_per_pair,
            confusion_pairs=confusion_pairs,
        )
