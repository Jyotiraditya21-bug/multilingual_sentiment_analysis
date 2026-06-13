import os
import sys
import uvicorn
import gradio as gr

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.main import app as fastapi_app
from frontend.app import demo as gradio_demo
import frontend.app as frontend_app

# Hugging Face Spaces automatically passes the 'PORT' environment variable (usually 7860)
port = int(os.environ.get("PORT", 7860))

# Configure Gradio API target to call the locally hosted FastAPI endpoints in the same process
frontend_app.API_URL = f"http://127.0.0.1:{port}"
print(f"Unified app routing frontend requests to backend API: {frontend_app.API_URL}")

# Remove FastAPI's root route ("/") so that the mounted Gradio app at "/" is served properly
fastapi_app.router.routes = [r for r in fastapi_app.router.routes if r.path != "/"]

# Mount Gradio dashboard at the root path "/" of the FastAPI application
app = gr.mount_gradio_app(fastapi_app, gradio_demo, path="/")

if __name__ == "__main__":
    print(f"Starting unified FastAPI + Gradio server on port {port}...")
    uvicorn.run("app_unified:app", host="0.0.0.0", port=port)
