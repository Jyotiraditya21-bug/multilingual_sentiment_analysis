"""FastAPI application for Multilingual Sentiment Analysis.

Exposes endpoints for single text analysis (with SHAP explanation) and
batch text classification. Loads the SentimentExplainer at startup.
"""

import os
import sys
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure explainability package can be imported from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explainability.shap_explain import SentimentExplainer

# Initialize FastAPI
app = FastAPI(
    title="Multilingual Sentiment Analysis API",
    description="Sentiment Analysis API using fine-tuned xlm-roberta-base with SHAP explainability.",
    version="1.0.0",
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global explainer instance
explainer: Optional[SentimentExplainer] = None
checkpoint_path = "model/checkpoints/best_model/"

# Fallback to Hugging Face model repository if local checkpoint is not found or incomplete
if not os.path.exists(checkpoint_path) or not os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
    checkpoint_path = "jimmy2110/multilingual-sentiment-model"


@app.on_event("startup")
def startup_event():
    """Startup lifecycle hook. Loads SentimentExplainer model."""
    global explainer
    print("FastAPI server starting up...")

    is_hf_model = "/" in checkpoint_path and not os.path.exists(checkpoint_path)
    if not is_hf_model and not os.path.exists(checkpoint_path):
        print(
            f"WARNING: Checkpoint path '{checkpoint_path}' does not exist yet. "
            "Please train the model by running 'python model/train.py' before making predictions."
        )
    else:
        try:
            print(f"Loading SentimentExplainer from: {checkpoint_path}")
            explainer = SentimentExplainer(checkpoint_path)
            print("SentimentExplainer loaded successfully!")
        except Exception as e:
            print(f"ERROR: Failed to load SentimentExplainer: {e}")


# Pydantic schemas
class PredictRequest(BaseModel):
    """Single prediction request schema."""

    text: str = Field(..., min_length=1, description="Text review to analyze")
    language: Optional[str] = Field(
        None, description="Language of the text (e.g. 'en', 'es')"
    )


class PredictResponse(BaseModel):
    """Single prediction response schema with SHAP scores."""

    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., description="Model confidence score")
    shap_scores: Dict[str, float] = Field(
        ..., description="Aggregated word-level SHAP explanation scores"
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request schema."""

    texts: List[str] = Field(
        ..., min_items=1, description="List of text reviews to classify"
    )


class BatchPredictResponseItem(BaseModel):
    """Single item response in batch prediction."""

    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., description="Model confidence score")


# Endpoints
@app.get("/")
def read_root():
    """Index metadata endpoint."""
    return {
        "status": "ok",
        "model": "xlm-roberta-base",
        "languages": ["en", "es"],
        "endpoints": ["/predict", "/predict/batch", "/health"],
    }


@app.get("/health")
def health_check():
    """Health check endpoint. Checks if model is loaded."""
    if explainer is None:
        return {
            "status": "uninitialized",
            "message": "Model checkpoint not loaded. Train model to initialize.",
            "initialized": False,
        }
    return {
        "status": "healthy",
        "model": "xlm-roberta-base",
        "initialized": True,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    """Predict sentiment of a single review text and return word SHAP scores.

    Args:
        request: PredictRequest body.

    Returns:
        PredictResponse JSON.
    """
    global explainer
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is uninitialized. Train the model first.",
        )

    # Empty text check
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text cannot contain only whitespaces.",
        )

    try:
        explanation = explainer.explain(request.text)
        return PredictResponse(
            label=explanation["label"],
            confidence=explanation["confidence"],
            shap_scores=explanation["shap_scores"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference prediction error: {str(e)}",
        )


@app.post("/predict/batch", response_model=List[BatchPredictResponseItem])
def predict_sentiment_batch(request: BatchPredictRequest):
    """Predict sentiment of batch reviews without generating SHAP explanations (for speed).

    Args:
        request: BatchPredictRequest body.

    Returns:
        List of BatchPredictResponseItem.
    """
    global explainer
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is uninitialized. Train the model first.",
        )

    # Validate inputs are not all empty
    for t in request.texts:
        if not t.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Batch texts cannot contain empty string elements.",
            )

    try:
        # Batch tokenize
        inputs = explainer.tokenizer(
            request.texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(explainer.device)

        with torch.no_grad():
            outputs = explainer.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()

        results = []
        for idx, conf in zip(pred_indices, confidences):
            results.append(
                BatchPredictResponseItem(
                    label=explainer.label_map[idx], confidence=float(conf)
                )
            )

        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference batch prediction error: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9090, reload=True)
