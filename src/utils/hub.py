from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()

def push_checkpoint_to_hub(checkpoint_path: str, repo_id: str) -> None:
    """Push model checkpoint directory to HuggingFace Hub."""

    # get the HF token from .env

    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")
    
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    api.upload_folder(
        folder_path=checkpoint_path,
        repo_id=repo_id,
        token=token,
        commit_message="Add model checkpoint"
    )