"""Training script for Multilingual Sentiment Analysis.

Fine-tunes xlm-roberta-base on preprocessed Amazon English & Spanish reviews.
Supports mixed-precision training, early stopping, and logs train/val metrics.
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Ensure data package can be imported from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import get_dataloaders


def evaluate(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate model on a dataloader.

    Args:
        model: PyTorch model.
        dataloader: Validation or test DataLoader.
        device: CPU/CUDA/MPS device.

    Returns:
        Tuple of (average_loss, accuracy, macro_f1).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1


def train(args: argparse.Namespace) -> None:
    """Train loop execution.

    Args:
        args: Parsed argparse CLI parameters.
    """
    # Setup directories
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints/best_model/")
    os.makedirs(checkpoint_dir, exist_ok=True)
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

    # Load DataLoaders
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        sample_size_per_lang=args.sample_size_per_lang,
        lowercase=args.lowercase,
        output_dir=artifacts_dir,
    )

    # Load Model
    print(f"Loading pre-trained model: {args.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3
    )
    if hasattr(model, "roberta"):
        print("Freezing the base RoBERTa encoder layers...")
        for param in model.roberta.parameters():
            param.requires_grad = False
    model.to(device)

    # Setup optimizer and scheduler (only optimize parameters that require gradients)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.10)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Mixed precision setup (GradScaler is only fully supported for CUDA devices)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (AMP) enabled: {use_amp}")

    # Early stopping config
    best_val_f1 = 0.0
    patience_counter = 0
    patience = args.patience

    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"
        )
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Conditional AMP autocast
            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=use_amp,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        print(
            f"\nEpoch {epoch + 1} Metrics: "
            f"Train Loss = {avg_train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f} | "
            f"Val Acc = {val_acc:.4f} | "
            f"Val Macro-F1 = {val_f1:.4f}"
        )

        # Early Stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best checkpoint
            model.save_pretrained(checkpoint_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.save_pretrained(checkpoint_dir)
            print(
                f"--> New best model checkpoint saved (Val F1: {best_val_f1:.4f})"
            )
        else:
            patience_counter += 1
            print(
                f"--> Validation F1 did not improve. Early stopping counter: {patience_counter}/{patience}"
            )
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Load best model for testing
    print("\nLoading best model checkpoint for final test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir
    )
    best_model.to(device)

    _, test_acc, test_f1 = evaluate(best_model, test_loader, device)

    # Print summary
    print("\n==================================================")
    print("               TRAINING SUMMARY                   ")
    print("==================================================")
    print(f"Best Val Macro-F1: {best_val_f1:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Macro-F1: {test_f1:.4f}")
    print(f"Best Checkpoint Path: {os.path.abspath(checkpoint_dir)}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa for Multilingual Sentiment Analysis (English & Spanish)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="xlm-roberta-base",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate for AdamW"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
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
        "--patience",
        type=int,
        default=3,
        help="Early stopping validation F1 patience",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/",
        help="Output directory to save model and tokenizer checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cpu, mps, cuda)",
    )
    cli_args = parser.parse_args()

    train(cli_args)
