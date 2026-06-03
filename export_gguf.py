import os
import argparse
import logging
import subprocess
import sys
from convert import convert_nano_to_hf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def export_to_gguf(checkpoint_path: str, hf_ref_path: str, output_gguf_path: str, llama_cpp_path: str = None):
    # Step 1: Export nano-llm to Hugging Face format
    temp_hf_dir = "./outputs/temp_hf_export"
    logger.info(f"Step 1/2: Exporting nano-llm model to Hugging Face format at {temp_hf_dir} ...")
    convert_nano_to_hf(checkpoint_path, hf_ref_path, temp_hf_dir)
    
    # Step 2: Convert Hugging Face directory to GGUF
    logger.info("Step 2/2: Converting Hugging Face model to llama.cpp GGUF format ...")
    
    # Check if we have llama.cpp path specified or if it's cloned locally
    if not llama_cpp_path:
        # Search for llama.cpp in common locations
        possible_paths = ["../llama.cpp", "./llama.cpp", "../llama-cpp", "./llama-cpp"]
        for p in possible_paths:
            if os.path.exists(os.path.join(p, "convert_hf_to_gguf.py")):
                llama_cpp_path = p
                break
                
    if llama_cpp_path and os.path.exists(os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")):
        converter_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        logger.info(f"Found llama.cpp converter script at: {converter_script}")
        
        # Command: python convert_hf_to_gguf.py temp_hf_dir --outfile output_gguf_path
        cmd = [
            sys.executable,
            converter_script,
            temp_hf_dir,
            "--outfile",
            output_gguf_path
        ]
        
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info(f"GGUF conversion completed successfully! Output saved to: {output_gguf_path}")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running llama.cpp converter script: {e}")
            logger.info("Falling back to manual instruction...")
    else:
        logger.warning("llama.cpp repository/converter script not found.")
        
    logger.info("=" * 80)
    logger.info("Hugging Face model successfully prepared at: " + os.path.abspath(temp_hf_dir))
    logger.info("To complete GGUF conversion, please follow these standard llama.cpp instructions:")
    logger.info("1. Clone llama.cpp repository:")
    logger.info("   git clone https://github.com/ggerganov/llama.cpp.git")
    logger.info("2. Install dependencies:")
    logger.info("   pip install -r llama.cpp/requirements.txt")
    logger.info("3. Convert to GGUF format (e.g. Q4_K_M quantization):")
    logger.info(f"   python llama.cpp/convert_hf_to_gguf.py {temp_hf_dir} --outfile {output_gguf_path}")
    logger.info("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: Llama.cpp GGUF Exporter")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to nano-llm .pt checkpoint file")
    parser.add_argument("--ref_hf", type=str, default="qwen/Qwen2.5-7B", help="Reference HF model for configurations")
    parser.add_argument("--output_gguf_path", type=str, required=True, help="Target GGUF file path (e.g., outputs/model.gguf)")
    parser.add_argument("--llama_cpp_path", type=str, default=None, help="Path to llama.cpp repository directory (optional)")
    args = parser.parse_args()
    
    export_to_gguf(args.checkpoint_path, args.ref_hf, args.output_gguf_path, args.llama_cpp_path)
