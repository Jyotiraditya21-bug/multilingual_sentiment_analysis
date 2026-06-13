# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /code

# Install system dependencies (required for building scientific packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ./requirements.txt /code/requirements.txt

# Optimize PyTorch installation: Use CPU-only wheels to save container size and speed up builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy the entire workspace code and checkpoints to the container
COPY . /code

# Set directory permissions for Hugging Face Spaces non-root user environment (UID 1000)
RUN chmod -R 777 /code

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

# Run the unified FastAPI + Gradio server
CMD ["python", "app_unified.py"]
