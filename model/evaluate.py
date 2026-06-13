"""Evaluation script for Multilingual Sentiment Analysis.

Loads the best trained checkpoint and evaluates on the test split.
Generates metrics and visualization plots, and saves them to assets/.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

# Ensure data package can be imported from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import get_dataloaders
from transformers import AutoModelForSequenceClassification


def evaluate_test(args: argparse.Namespace) -> None:
    """Run evaluation on the test set and output visualizations/reports.

    Args:
        args: Parsed argparse CLI parameters.
    """
    # Create target directories
    assets_dir = args.assets_dir
    os.makedirs(assets_dir, exist_ok=True)
    artifacts_dir = os.path.join(args.output_dir, "artifacts/")

    # Setup device
    if getattr(args, "device", None) is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    # Load dataloaders
    print("Loading test dataset...")
    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        sample_size_per_lang=args.sample_size_per_lang,
        lowercase=args.lowercase,
        output_dir=artifacts_dir,
    )

    # Load model
    print(f"Loading best checkpoint from: {args.checkpoint}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    # Collect predictions and labels
    all_probs = []
    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = ["Negative", "Neutral", "Positive"]
    n_classes = len(class_names)

    # 1. Classification Report
    print("\nClassification Report:")
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names
    )
    print(report_str)

    report_path = os.path.join(assets_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"Saved classification report to: {report_path}")

    # 2. Confusion Matrix Heatmap
    print("\nGenerating confusion matrix heatmap...")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.ylabel("Actual Label", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.title("Confusion Matrix Heatmap", fontsize=14, pad=15)
    plt.tight_layout()

    cm_path = os.path.join(assets_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 3. AUC-ROC curve (one-vs-rest)
    print("\nGenerating One-vs-Rest AUC-ROC curves...")
    # Binarize labels for multi-class ROC calculation
    y_bin = label_binarize(all_labels, classes=[0, 1, 2])

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(9, 7))
    colors = ["#ff4d4d", "#3399ff", "#2eb82e"]  # Red, Blue, Green colors

    print("\nPer-class AUC Scores:")
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f" - {class_names[i]}: AUC = {roc_auc[i]:.4f}")

        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.4f})",
        )

    # Plot baseline
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12, labelpad=10)
    plt.title("AUC-ROC Curves (One-vs-Rest)", fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(assets_dir, "auc_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved AUC-ROC curve plot to: {roc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned XLM-RoBERTa model on English and Spanish"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_model/",
        help="Path to best checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/",
        help="Directory where preprocess artifacts are located",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="assets/",
        help="Directory to save evaluation charts and reports",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument(
        "--max_len", type=int, default=128, help="Tokenization max length"
    )
    parser.add_argument(
        "--sample_size_per_lang",
        type=int,
        default=50000,
        help="Subsample size per language subset",
    )
    parser.add_argument(
        "--lowercase", action="store_true", help="Lowercase raw text"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on (cpu, mps, cuda)",
    )
    cli_args = parser.parse_args()

    evaluate_test(cli_args)
