"""SHAP explanation script for Multilingual Sentiment Analysis.

Defines the SentimentExplainer class which wraps the model and tokenizer,
computes SHAP token importances, and aggregates subwords to whole words.
"""

import os
import sys
from typing import Dict, List, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Ensure shap is imported safely
try:
    import shap
except ImportError:
    raise ImportError("Please run 'pip install shap' before running this script.")


class SentimentExplainer:
    """Explainer class for Multilingual Sentiment Analysis using SHAP."""

    def __init__(self, checkpoint_path: str):
        """Initialize Explainer.

        Loads model, tokenizer, and sets up SHAP Explainer.

        Args:
            checkpoint_path: Path to model checkpoint.
        """
        self.checkpoint_path = checkpoint_path

        # Determine device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"SentimentExplainer using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # Load model: check if we load from base backbone + classifier head
        head_path = os.path.join(checkpoint_path, "classifier_head.pt")
        if os.path.exists(head_path):
            print(f"Loading base backbone xlm-roberta-base and custom classification head from: {head_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "FacebookAI/xlm-roberta-base", num_labels=3
            )
            # Load custom classification head weights
            self.model.classifier.load_state_dict(torch.load(head_path, map_location=self.device))
        else:
            print(f"Loading full model from checkpoint: {checkpoint_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path
            )
        self.model.to(self.device)
        self.model.eval()

        # Label mappings
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

        # Define transformers pipeline for SHAP
        # Note: pipeline requires device mapping as int or torch.device
        pipeline_device = 0 if self.device.type == "cuda" else -1
        self.pred_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
            top_k=None,  # Return probabilities for all classes
        )

        # Initialize SHAP explainer
        # Using the standard text partition explainer
        self.explainer = shap.Explainer(self.pred_pipeline)

    def aggregate_subwords(
        self, tokens: List[str], shap_values: List[float]
    ) -> Dict[str, float]:
        """Aggregate subword token SHAP scores into whole words.

        Args:
            tokens: List of subwords/tokens from tokenizer.
            shap_values: Raw SHAP scores of each token.

        Returns:
            Dictionary of whole words mapped to aggregated SHAP values.
        """
        word_scores = {}
        current_word = ""
        current_score = 0.0
        prev_ended_with_space = True

        for token, val in zip(tokens, shap_values):
            if not token or token in ["<s>", "</s>", "<pad>", "<unk>"]:
                continue

            # Clean SentencePiece block character and standard space
            token_clean = token.replace(" ", "").replace(" ", "")

            # A new word starts if the previous token ended in space, or the current token starts with space
            starts_new = prev_ended_with_space or token.startswith(" ") or token.startswith(" ")

            if starts_new and current_word:
                word_scores[current_word] = (
                    word_scores.get(current_word, 0.0) + current_score
                )
                current_word = token_clean
                current_score = val
            else:
                if not current_word:
                    current_word = token_clean
                    current_score = val
                else:
                    current_word += token_clean
                    current_score += val

            prev_ended_with_space = token.endswith(" ") or token.endswith(" ")

        if current_word:
            word_scores[current_word] = (
                word_scores.get(current_word, 0.0) + current_score
            )

        return word_scores

    def explain(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """Explain the sentiment prediction of a text sample.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary containing predicted label, confidence, and aggregated SHAP scores.
        """
        if not text.strip():
            return {"label": "neutral", "confidence": 0.0, "shap_scores": {}}

        # 1. Run direct inference to get label and confidence
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        pred_label = self.label_map[pred_idx]

        # 2. Run SHAP explainer
        shap_values = self.explainer([text])
        explanation = shap_values[0]

        # Get class dimension based on predicted label/config
        # XLM-RoBERTa standard pipeline output labels are "LABEL_0", etc.
        class_name = self.model.config.id2label.get(pred_idx, f"LABEL_{pred_idx}")

        if (
            hasattr(explanation, "output_names")
            and explanation.output_names is not None
        ):
            if class_name in explanation.output_names:
                class_dim = explanation.output_names.index(class_name)
            else:
                class_dim = pred_idx
        else:
            class_dim = pred_idx

        # Extract values
        raw_scores = explanation.values[:, class_dim].tolist()
        tokens = explanation.data.tolist()

        # 3. Aggregate subwords to whole words
        aggregated_scores = self.aggregate_subwords(tokens, raw_scores)

        return {
            "label": pred_label,
            "confidence": float(confidence),
            "shap_scores": aggregated_scores,
        }


if __name__ == "__main__":
    print("--- SHAP EXPLAINER DEMO ---")
    # Check if best model checkpoint exists
    checkpoint = "model/checkpoints/best_model/"
    if not os.path.exists(checkpoint):
        print(
            f"Checkpoint directory '{checkpoint}' not found. "
            "Please train the model first by running: python model/train.py"
        )
        sys.exit(1)

    explainer = SentimentExplainer(checkpoint)

    # 3 example reviews: English positive, Spanish negative, English neutral
    examples = [
        "This product is amazing! I highly recommend it.",  # EN Positive
        "Producto muy malo, una pérdida de dinero.",  # ES Negative
        "It is okay, not good not bad.",  # EN Neutral
    ]

    for ex in examples:
        print("\n" + "=" * 50)
        print(f"Review: {ex}")
        print("=" * 50)
        result = explainer.explain(ex)
        print(f"Predicted Sentiment: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Aggregated SHAP Scores:")
        for word, val in result["shap_scores"].items():
            print(f"  {word:15s} : {val:+.4f}")
