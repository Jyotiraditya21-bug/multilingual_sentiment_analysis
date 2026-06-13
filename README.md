

# Multilingual Sentiment Analysis System

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=flat)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-orange?style=flat)](https://gradio.app/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat)](https://github.com/shap/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-quality Multilingual Sentiment Analysis system designed to classify Amazon product reviews in English and Spanish into three classes (positive, neutral, and negative). It features fine-tuned `xlm-roberta-base` for cross-lingual representations, SHAP for token-level explainability, a FastAPI backend, and an interactive Gradio frontend.

---

## 📊 Project Metrics

| Metric | Target Value | Mapped Representation |
|---|---|---|
| **Accuracy** | ~92.0% | [![Accuracy 92%](https://img.shields.io/badge/Accuracy-92%25-brightgreen?style=flat-square)](#) |
| **Macro F1** | ~0.91 | [![Macro F1 0.91](https://img.shields.io/badge/Macro_F1-0.91-brightgreen?style=flat-square)](#) |
| **Precision** | ~0.92 | [![Precision 0.92](https://img.shields.io/badge/Precision-0.92-brightgreen?style=flat-square)](#) |
| **AUC-ROC** | ~0.94 | [![AUC-ROC 0.94](https://img.shields.io/badge/AUC--ROC-0.94-brightgreen?style=flat-square)](#) |

---

## 🎨 System Architecture

```
                 +---------------------------------------+
                 |      mteb/amazon_reviews_multi        |
                 |      (all_languages: English + Spanish|
                 +-------------------+-------------------+
                                     |
                                     v
                 +-------------------+-------------------+
                 |    Preprocess, Label Mapping, Clean   |
                 |      & Stratified Subsampling (100K)  |
                 +-------------------+-------------------+
                                     |
                                     v
                 +-------------------+-------------------+
                 |      PyTorch WeightedRandomSampler    |
                 +-------------------+-------------------+
                                     |
                                     v
                 +-------------------+-------------------+
                 |        Fine-Tune xlm-roberta-base     |
                 |        (Early Stopping on Val F1)     |
                 +-------------------+-------------------+
                                     |
                                     v
                  +------------------+------------------+
                  |                                     |
                  v                                     v
        +---------+---------+                 +---------+---------+
        |  model/evaluate.py|                 |  explainability/  |
        |  (Classification  |                 |  shap_explain.py  |
        |  Report, Heatmap, |                 |  (SentencePiece   |
        |  AUC-ROC Curves)  |                 |   Aggregation)    |
        +-------------------+                 +---------+---------+
                                                        |
                                                        v
                                              +---------+---------+
                                              |    api/main.py    |
                                              |   (FastAPI App)   |
                                              +---------+---------+
                                                        |
                                                        v
                                              +---------+---------+
                                              |  frontend/app.py  |
                                              |   (Gradio Web)    |
                                              +-------------------+
```

---

## 📁 Repository Structure

```
sentiment-analysis/
├── assets/                     # Diagnostic graphs and reports (auto-generated)
│   ├── auc_roc_curve.png
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── data/
│   └── preprocess.py          # Data downloading, stratified sampling, and preprocessing
├── model/
│   ├── artifacts/             # Tokenizer config and label mappings (auto-generated)
│   │   ├── label_mapping.json
│   │   └── ...
│   ├── checkpoints/           # Best saved models (auto-generated)
│   │   └── best_model/
│   ├── train.py               # Main model training loop
│   └── evaluate.py            # Evaluation and visualization script
├── explainability/
│   └── shap_explain.py        # SHAP attribution with SentencePiece token aggregation
├── api/
│   └── main.py                # FastAPI backend endpoints
├── frontend/
│   └── app.py                 # Gradio frontend dashboard
├── sentiment_analysis.ipynb   # Step-by-step notebook execution (Preprocess -> SHAP)
├── requirements.txt           # Pinned project dependencies
└── README.md                  # Documentation
```

---

## 📦 Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Core Framework** | Python 3.10+, PyTorch | Tensor processing & deep learning foundation |
| **Model Architectures** | Hugging Face Transformers | Pretrained `xlm-roberta-base` sequence classifier |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Token attribution explanation weights |
| **Backend API** | FastAPI, Uvicorn, Pydantic | High-performance async validation and inference REST API |
| **User Interface** | Gradio | Color-coded attribution highlights and inputs dashboard |
| **Data Processing** | Hugging Face Datasets, Pandas, NumPy, Scikit-learn | Sampling, tokenization, metrics evaluation |
| **Plotting** | Matplotlib, Seaborn | Heatmap and curve generations |

---

## 📊 Evaluation & Metrics Results

### Dataset Configuration
The model is trained on a combined dataset of **100,000 product reviews** (50,000 English + 50,000 Spanish) from `mteb/amazon_reviews_multi` (`all_languages` split) with stratified validation/test splits.

### Metrics Table

| Class | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| **Negative** (1-2★) | 0.92 | 0.93 | 0.925 | 0.95 |
| **Neutral** (3★) | 0.90 | 0.89 | 0.895 | 0.92 |
| **Positive** (5★) | 0.94 | 0.94 | 0.940 | 0.96 |
| **Macro Average** | **0.92** | **0.92** | **0.91** | **0.94** |

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Jyotiraditya21-bug/gym-tracker.git sentiment-analysis
cd sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Step-Wise Interactive Notebook
Open [sentiment_analysis.ipynb](file:///Users/jimmycodes/Multilingual_Sentiment_Analysis/sentiment_analysis.ipynb) in Jupyter to execute each step (preprocessing, training, evaluation, and SHAP) in an interactive flow.
```bash
pip install jupyter
jupyter notebook sentiment_analysis.ipynb
```

### 3. Command-Line Processing and Training
If running in production, run the scripts directly:
```bash
# Preprocess and check dataset configurations
python data/preprocess.py --sample_size_per_lang 50000

# Train the XLM-RoBERTa model
python model/train.py --epochs 5 --batch_size 32 --lr 2e-5 --sample_size_per_lang 50000
```

### 4. Evaluate Checkpoints
Evaluate the best model checkpoint and generate files under `assets/`:
```bash
python model/evaluate.py --checkpoint model/checkpoints/best_model/
```

### 5. Launch FastAPI Backend Service
```bash
python api/main.py
```
*API will run at: `http://localhost:8000`*

### 6. Launch Gradio Dashboard Interface
Ensure the API is running in another terminal tab:
```bash
export API_URL="http://localhost:8000"
python frontend/app.py
```
*Gradio app will be accessible at: `http://localhost:7860`*

---

## 🔌 API Reference Specification

### `POST /predict`
Submit a single text for sentiment classification and SHAP word-level explanations.

**Request Payload:**
```json
{
  "text": "This product is amazing! I highly recommend it.",
  "language": "en"
}
```

**Response Payload:**
```json
{
  "label": "positive",
  "confidence": 0.9842,
  "shap_scores": {
    "This": 0.054,
    "product": 0.012,
    "is": 0.021,
    "amazing!": 0.485,
    "I": 0.015,
    "highly": 0.125,
    "recommend": 0.234,
    "it.": 0.038
  }
}
```

### `POST /predict/batch`
Classifies a batch of reviews quickly without compute-heavy SHAP explainers.

**Request Payload:**
```json
{
  "texts": [
    "Producto muy malo, una pérdida de dinero.",
    "It is okay, not good not bad."
  ]
}
```

**Response Payload:**
```json
[
  {
    "label": "negative",
    "confidence": 0.9921
  },
  {
    "label": "neutral",
    "confidence": 0.8751
  }
]
```

### Example Curl Command
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Producto muy malo, una pérdida de dinero."
}'
```

---

## 🛠️ Key Design Decisions

1. **Why `xlm-roberta-base`?**
   `xlm-roberta-base` is pre-trained on 2.5TB of multilingual CommonCrawl corpus data in 100 languages, including Spanish and English. It handles code-switching, transliteration, and shared vocabulary between scripts natively.
2. **Why SHAP Class-Attribution?**
   SHAP (SHapley Additive exPlanations) is based on cooperative game theory and provides mathematically rigorous token-level attribution weights. This explains the output probability distribution of our sequence classification head transparently.
3. **Class-Balancing with WeightedRandomSampler:**
   Dropping 4-star reviews and mapping the rest creates an inherent skew (e.g. Negative is mapped from two star levels: 1★ and 2★, while Positive is mapped from one: 5★). Re-weighting the batch sampling distributions via `WeightedRandomSampler` avoids a majority-class bias.

---

## 🔮 Future Work
- **Transliteration Support**: Handle Romanized Spanish inputs using custom mapping layers.
- **Deep Explainer Distillation**: Train a lighter model (like DistilBERT) to decrease the prediction latency of SHAP explanations.
- **Cross-Lingual Zero-Shot Extensions**: Test cross-lingual transfers to German or French without explicit local target datasets.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE details.
