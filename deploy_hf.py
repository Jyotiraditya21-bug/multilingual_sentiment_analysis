import os
import sys
from huggingface_hub import HfApi

def deploy():
    print("==================================================")
    print("    HUGGING FACE SPACES DEPLOYMENT ASSISTANT     ")
    print("==================================================")
    print("\nThis script will upload the codebase and model checkpoints")
    print("directly to your Hugging Face Space using the Python Hub API.")
    print("Large files (like the 1.1GB model weights) will be uploaded as LFS automatically.")

    # 1. Prompt for inputs
    repo_id = input("\nEnter your Space Repo ID (e.g. username/sentiment-dashboard): ").strip()
    if not repo_id or "/" not in repo_id:
        print("ERROR: Repo ID must be in the format 'username/space-name'.")
        return

    hf_token = input("Enter your Hugging Face Access Token (with WRITE permission): ").strip()
    if not hf_token:
        print("ERROR: Token cannot be empty. Get it from https://huggingface.co/settings/tokens")
        return

    # 2. Confirm and execute upload
    print(f"\nStaging files to Space: '{repo_id}'...")
    print("Uploading folders (ignoring virtual environment and caches)...")

    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                "venv/**",
                ".venv/**",
                "env/**",
                "**/.git/**",
                "**/.ipynb_checkpoints/**",
                "**/.system_generated/**",
                "**/.tempmediaStorage/**",
                "**/*.pyc",
                "**/*.pyo",
                "**/__pycache__/**",
                "deploy_hf.py",  # Exclude script to prevent token exposure
            ],
            token=hf_token
        )
        print("\n==================================================")
        print("            UPLOAD COMPLETED SUCCESSFULLY!        ")
        print("==================================================")
        print(f"\nYour Space is building at: https://huggingface.co/spaces/{repo_id}")
        print("Wait a few minutes for the container build to finish.")
        print("==================================================")
    except Exception as e:
        print(f"\nERROR during deployment: {e}")

if __name__ == "__main__":
    deploy()
