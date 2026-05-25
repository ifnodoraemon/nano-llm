import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Unified Hub Uploader: Hugging Face & ModelScope Checkpoint Publisher
# ==============================================================================

def upload_to_huggingface(folder_path: str, repo_id: str, token: str = None):
    """
    Publishes local model files and checkpoints to the Hugging Face Hub.
    """
    logger.info("Initializing Hugging Face Hub API connection...")
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        # Verify repo_id format (must be username/repo_name)
        if "/" not in repo_id:
            logger.error("Hugging Face repo_id must be in the format 'username/repo_name'!")
            return
            
        logger.info(f"Uploading files from '{folder_path}' to HF Hub repository: '{repo_id}'...")
        
        # Attempt to create the repository if it does not exist
        try:
            api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
            logger.info(f"✅ Target repository '{repo_id}' is ready on Hugging Face.")
        except Exception as e:
            logger.warning(f"Unable to verify/create repository on HF (proceeding anyway): {e}")

        # Upload the entire folder containing checkpoints and logs
        future = api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        logger.info("=======================================================================")
        logger.info(f"✅ Successfully published checkpoint weights to Hugging Face Hub!")
        logger.info(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
        logger.info("=======================================================================")
        
    except ImportError:
        logger.error("The 'huggingface_hub' library is not installed.")
        logger.info("To fix this, please run: pip install huggingface_hub")
    except Exception as e:
        logger.error(f"Hugging Face upload failed with error: {e}")


def upload_to_modelscope(folder_path: str, repo_id: str, token: str = None):
    """
    Publishes local model files and checkpoints to the ModelScope Hub (Alibaba Group).
    Highly recommended for high-speed download speeds on GPU clusters within China.
    """
    logger.info("Initializing ModelScope Hub API connection...")
    try:
        from modelscope.hub.api import HubApi
        
        # Verify repo_id format (must be username/repo_name)
        if "/" not in repo_id:
            logger.error("ModelScope repo_id must be in the format 'username/repo_name'!")
            return
            
        api = HubApi()
        
        # Authenticate using SDK token if provided
        if token:
            logger.info("Authenticating with ModelScope SDK Token...")
            api.login(token)
            
        logger.info(f"Uploading files from '{folder_path}' to ModelScope Hub repository: '{repo_id}'...")
        
        # Attempt to create the repository if it does not exist
        try:
            # ModelScope create_repo has different parameters
            # Under ModelScope, model ID format is username/repo_name
            api.create_model(model_id=repo_id, visibility=1, exist_ok=True) # visibility=1 represents Public
            logger.info(f"✅ Target repository '{repo_id}' is ready on ModelScope.")
        except Exception as e:
            logger.warning(f"Unable to verify/create repository on ModelScope (proceeding anyway): {e}")

        # Upload the entire directory to ModelScope
        api.push_model(
            model_id=repo_id,
            model_dir=folder_path,
            commit_message="feat: upload nano-llm model weights and config checkpoint"
        )
        
        logger.info("=======================================================================")
        logger.info(f"✅ Successfully published checkpoint weights to ModelScope Hub!")
        logger.info(f"🔗 Repository URL: https://modelscope.cn/models/{repo_id}")
        logger.info("=======================================================================")
        
    except ImportError:
        logger.error("The 'modelscope' library is not installed.")
        logger.info("To fix this, please run: pip install modelscope")
    except Exception as e:
        logger.error(f"ModelScope upload failed with error: {e}")


def main():
    parser = argparse.ArgumentParser(description="nano-llm: Unified Hugging Face & ModelScope Hub Checkpoint Publisher")
    parser.add_argument("--hub", type=str, choices=["both", "hf", "modelscope"], default="both", help="Target platform to publish weights")
    parser.add_argument("--folder", type=str, default="./outputs", help="Local directory containing model checkpoints to upload")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository destination ID (format: username/repo_name)")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face Hub Write Token")
    parser.add_argument("--ms_token", type=str, default=os.environ.get("MODELSCOPE_TOKEN"), help="ModelScope SDK Access Token")
    args = parser.parse_args()

    # Validate local directory existence
    if not os.path.exists(args.folder):
        logger.error(f"The local folder '{args.folder}' does not exist!")
        return

    # Trigger uploads
    if args.hub in ["both", "hf"]:
        upload_to_huggingface(folder_path=args.folder, repo_id=args.repo_id, token=args.hf_token)
        
    if args.hub in ["both", "modelscope"]:
        upload_to_modelscope(folder_path=args.folder, repo_id=args.repo_id, token=args.ms_token)

if __name__ == "__main__":
    main()
