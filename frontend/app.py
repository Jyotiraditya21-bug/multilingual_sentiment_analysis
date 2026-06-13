"""Gradio Web Application for Multilingual Sentiment Analysis.

Connects to the FastAPI backend to perform sentiment prediction and display
word-level SHAP explanation highlights for English and Spanish text reviews.
"""

import os
import string
from typing import Dict, List, Tuple

import gradio as gr
import requests

# Fetch backend API URL from environment variables
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


def clean_and_map_shap(text: str, shap_scores: Dict[str, float]) -> List[Tuple[str, float]]:
    """Map raw SHAP scores back to individual words in the original text order.

    Args:
        text: Input review text.
        shap_scores: Map of aggregated word strings to SHAP scores.

    Returns:
        List of tuples formatted for gr.HighlightedText.
    """
    words = text.split()
    highlights = []

    for word in words:
        # Standard cleanups to match SHAP words
        word_clean = word.strip(".,!?;:()\"'[]{}|।")
        score = 0.0

        if word in shap_scores:
            score = shap_scores[word]
        elif word_clean in shap_scores:
            score = shap_scores[word_clean]
        elif word_clean.lower() in shap_scores:
            score = shap_scores[word_clean.lower()]
        else:
            # Fallback check for substrings
            for key, val in shap_scores.items():
                if (
                    key.lower() == word_clean.lower()
                    or key.lower() in word_clean.lower()
                    or word_clean.lower() in key.lower()
                ):
                    score = val
                    break

        highlights.append((word + " ", score))

    return highlights


def predict_sentiment(text: str, language: str) -> Tuple[Dict[str, float], List[Tuple[str, float]], Dict[str, float]]:
    """Query FastAPI backend for predictions and explanations.

    Args:
        text: Review text.
        language: Language string (English or Spanish).

    Returns:
        Tuple of (label_conf, highlighted_text, raw_json_scores).
    """
    if not text.strip():
        return {"neutral": 1.0}, [], {}

    # Map language label
    lang_code = "en" if language == "English" else "es"

    try:
        # Call backend api
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "language": lang_code},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )

        if response.status_code != 200:
            error_msg = f"API Error (Status {response.status_code}): {response.text}"
            print(error_msg)
            return {"error": 1.0}, [(error_msg, 0.0)], {"error": response.text}

        data = response.json()
        label = data["label"]
        confidence = data["confidence"]
        shap_scores = data["shap_scores"]

        # Convert label for gr.Label (needs dict of labels and probabilities)
        # To make a clean display, we populate other classes with small residual values
        label_probs = {
            "negative": 0.0,
            "neutral": 0.0,
            "positive": 0.0
        }
        label_probs[label] = confidence
        remaining = 1.0 - confidence
        other_labels = [k for k in label_probs.keys() if k != label]
        for ol in other_labels:
            label_probs[ol] = remaining / len(other_labels)

        # Generate highlights mapping
        highlights = clean_and_map_shap(text, shap_scores)

        return label_probs, highlights, shap_scores

    except requests.exceptions.ConnectionError:
        err_msg = (
            f"Failed to connect to FastAPI backend at {API_URL}. "
            "Please ensure the API server is running ('python api/main.py')."
        )
        print(err_msg)
        return {"backend_offline": 1.0}, [(err_msg, 0.0)], {"error": err_msg}
    except Exception as e:
        err_msg = f"Inference processing failed: {str(e)}"
        print(err_msg)
        return {"error": 1.0}, [(err_msg, 0.0)], {"error": err_msg}


# Custom CSS for UI styling
custom_css = """
/* Force dark mode backgrounds globally */
html, body, .gradio-container, .gradio-container-6-0-0, .gradio-container-6-0-1 {
    background-color: #090d16 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(20, 184, 166, 0.04) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(16, 185, 129, 0.02) 0px, transparent 50%) !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.container {
    max-width: 1200px;
    margin: auto;
    padding: 2rem 1.5rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.header h1 {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.header p {
    font-size: 1.15rem;
    color: #94a3b8;
    max-width: 650px;
    margin: auto;
}

/* KPI metric cards style */
.metric-container {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
    width: 100%;
}

.metric-card {
    flex: 1;
    min-width: 220px;
    background: rgba(18, 22, 32, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.metric-card:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(20, 184, 166, 0.25) !important;
    box-shadow: 0 8px 30px rgba(20, 184, 166, 0.08) !important;
}

.metric-title {
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #64748b !important;
    font-weight: 700 !important;
    margin-bottom: 0.25rem !important;
}

.metric-value {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}

.metric-value.silver {
    background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.teal {
    background: linear-gradient(135deg, #2dd4bf 0%, #0d9488 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.emerald {
    background: linear-gradient(135deg, #34d399 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-value.shap {
    background: linear-gradient(135deg, #f43f5e 0%, #be123c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-subtitle {
    font-size: 0.72rem !important;
    color: #475569 !important;
    margin-top: 0.25rem !important;
}

/* Card layout elements */
.input-card, .examples-card, .insights-card, .attribution-card {
    padding: 1.5rem !important;
    background: rgba(18, 22, 32, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
    margin-bottom: 1.5rem !important;
    transition: all 0.3s ease !important;
}

.input-card:hover, .examples-card:hover, .insights-card:hover, .attribution-card:hover {
    border-color: rgba(20, 184, 166, 0.2) !important;
    box-shadow: 0 8px 32px rgba(20, 184, 166, 0.04) !important;
}

.gradio-container h3 {
    margin-top: 0 !important;
    margin-bottom: 1rem !important;
    font-size: 1.05rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
}

textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

textarea:focus {
    border-color: #14b8a6 !important;
    box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.15) !important;
    background: rgba(15, 23, 42, 0.8) !important;
}

.submit-btn {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
    border: 1px solid #14b8a6 !important;
    color: #fff !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 4px 20px rgba(20, 184, 166, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-top: 1rem;
    border-radius: 8px !important;
}

.submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(20, 184, 166, 0.25) !important;
    background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%) !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #090d16;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}
"""

# Premium Custom Theme (Slate & Teal)
premium_theme = gr.themes.Default(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
).set(
    body_background_fill="#090d16",
    body_background_fill_dark="#090d16",
    block_background_fill="#121620",
    block_background_fill_dark="#121620",
    block_border_color="#1e293b",
    block_border_color_dark="#1e293b",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    input_background_fill="#1a202c",
    input_background_fill_dark="#1a202c",
    input_border_color="#2d3748",
    input_border_color_dark="#2d3748",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    button_primary_background_fill="linear-gradient(135deg, #0f766e 0%, #115e59 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #14b8a6 0%, #0f766e 100%)",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(title="Multilingual Sentiment Dashboard") as demo:
    # Force dark mode class on document load
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            document.documentElement.classList.add('dark');
            localStorage.setItem('color-theme', 'dark');
        }
        """
    )

    with gr.Group(elem_classes="container"):
        # Header Section
        with gr.Group(elem_classes="header"):
            gr.Markdown(
                """
                # Multilingual Sentiment Dashboard
                Executive portal for cross-lingual sentiment intelligence with mathematically rigorous token attribution explanations.
                """
            )

        # KPI Metrics Row
        gr.HTML(
            """
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-title">Classifier Model</div>
                    <div class="metric-value silver">XLM-RoBERTa</div>
                    <div class="metric-subtitle">Frozen Backbone</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Hardware Device</div>
                    <div class="metric-value teal">Apple Silicon MPS</div>
                    <div class="metric-subtitle">GPU Accelerated</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Languages Support</div>
                    <div class="metric-value emerald">English & Spanish</div>
                    <div class="metric-subtitle">Cross-lingual head</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Explainability</div>
                    <div class="metric-value shap">SHAP Attribution</div>
                    <div class="metric-subtitle">Word-Level Highlights</div>
                </div>
            </div>
            """
        )

        # Main Workspace Grid Layout
        with gr.Row():
            # Left Panel: Input controls
            left_panel = gr.Column(scale=2, min_width=350)
            with left_panel:
                with gr.Group(elem_classes="input-card"):
                    gr.Markdown("### Input Review")
                    text_input = gr.Textbox(
                        label="Review Text",
                        placeholder="Type review here... / Ingrese opinión aquí...",
                        lines=6,
                        max_lines=12,
                        show_label=False,
                    )
                    with gr.Row():
                        lang_input = gr.Radio(
                            choices=["English", "Spanish"],
                            value="English",
                            label="Language / Idioma",
                        )
                    submit_btn = gr.Button("Execute Analysis", elem_classes="submit-btn")
            
            # Right Panel: Live Insights & Attribution Reports
            right_panel = gr.Column(scale=3, min_width=450)
            with right_panel:
                with gr.Group(elem_classes="insights-card"):
                    gr.Markdown("### Analysis Report")
                    label_output = gr.Label(
                        num_top_classes=3,
                        show_label=False,
                    )
                
                with gr.Group(elem_classes="attribution-card"):
                    gr.Markdown("### SHAP Feature Attributions")
                    highlight_output = gr.HighlightedText(
                        show_label=False,
                        show_legend=True,
                        color_map={"+": "green", "-": "red"},
                    )

                with gr.Accordion("Raw SHAP Scores Map (Developer view)", open=False):
                    json_output = gr.JSON(show_label=False)

            # Re-enter left panel context to place Examples under inputs
            with left_panel:
                with gr.Group(elem_classes="examples-card"):
                    gr.Markdown("### Preset Review Benchmarks")
                    gr.Examples(
                        examples=[
                            ["This product is amazing! I highly recommend it.", "English"],
                            ["Producto muy malo, una pérdida de dinero.", "Spanish"],
                            ["It is okay, not good not bad.", "English"],
                        ],
                        inputs=[text_input, lang_input],
                        outputs=[label_output, highlight_output, json_output],
                        fn=predict_sentiment,
                        cache_examples=False,
                    )

        # Wiring functions
        submit_btn.click(
            fn=predict_sentiment,
            inputs=[text_input, lang_input],
            outputs=[label_output, highlight_output, json_output],
        )

# Launch parameters as specified in prompt
if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, theme=premium_theme, css=custom_css)
