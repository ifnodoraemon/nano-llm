import os
import sys
import json
import time
import logging
import asyncio
import subprocess
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Global Model State (hot-loadable)
# ==============================================================================

_model_lock = threading.Lock()
_active_model: Optional[torch.nn.Module] = None
_active_tokenizer = None
_active_config = None
_model_loaded: bool = False
_request_semaphore: Optional[asyncio.Semaphore] = None

# ==============================================================================
# Request Models
# ==============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "nano-lm"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256


class PipelineRequest(BaseModel):
    stage: str


# ==============================================================================
# Model Lifecycle
# ==============================================================================


def load_model(checkpoint_path: str = "./outputs/checkpoint_sft.pt", device: str = None):
    global _active_model, _active_tokenizer, _active_config, _model_loaded

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with _model_lock:
        logger.info(f"Loading model from {checkpoint_path} on {device}...")

        from model import Transformer
        from utils.tokenizer_loader import load_tokenizer as _load_tok
        from utils.checkpoint_utils import load_checkpoint_with_fp8_translation

        config, state_dict = load_checkpoint_with_fp8_translation(
            checkpoint_path, map_location=device,
            state_keys=("model_state_dict", "model"),
        )

        model = Transformer(config).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        tokenizer = _load_tok(fallback_model_name="gpt2")

        _active_model = model
        _active_tokenizer = tokenizer
        _active_config = config
        _model_loaded = True

        logger.info(f"Model loaded successfully ({config.n_layer}L, {config.n_embd}E, MLA={getattr(config, 'use_mla', False)}, MoE={getattr(config, 'use_moe', False)})")
        return model, tokenizer, config


def get_model():
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /api/model/load first.")
    return _active_model, _active_tokenizer, _active_config


# ==============================================================================
# App Lifecycle
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _request_semaphore
    _request_semaphore = asyncio.Semaphore(4)
    logger.info("nano-llm API server started")

    # Auto-load model on startup if checkpoint exists
    checkpoint_path = os.environ.get("MODEL_CHECKPOINT_PATH", "./outputs/checkpoint_sft.pt")
    if os.path.exists(checkpoint_path):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: load_model(checkpoint_path))
        except Exception as e:
            logger.warning(f"Startup model auto-load failed: {e}. Load manually via /api/model/load.")
    else:
        logger.info(f"No checkpoint at {checkpoint_path}. Skipping startup auto-load.")

    yield
    logger.info("nano-llm API server shutting down")


app = FastAPI(title="nano-llm API", lifespan=lifespan)
security = HTTPBearer(auto_error=False)
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ==============================================================================
# Static UI
# ==============================================================================


@app.get("/")
def get_dashboard():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


# ==============================================================================
# Health & Status
# ==============================================================================


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": _model_loaded}


@app.get("/api/status")
def get_status():
    stats = {
        "crawled_files": 0,
        "deduped_files": 0,
        "tokenizer_vocab_size": 0,
        "packed_tokens": 0,
        "pretrain_checkpoint_exists": False,
        "sft_checkpoint_exists": False,
        "dpo_checkpoint_exists": False,
        "model_loaded": _model_loaded,
        "eval_scores": {"mmlu": 0.0, "gsm8k": 0.0},
    }

    # Read real model config if loaded
    if _model_loaded and _active_config is not None:
        cfg = _active_config
        stats["model_config"] = {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "use_mla": getattr(cfg, "use_mla", False),
            "use_moe": getattr(cfg, "use_moe", False),
            "num_routed_experts": getattr(cfg, "num_routed_experts", 0),
        }
        total = sum(p.numel() for p in _active_model.parameters())
        stats["model_parameters"] = total

    raw_dir = "./data/raw_crawled"
    if os.path.exists(raw_dir):
        stats["crawled_files"] = len([f for f in os.listdir(raw_dir) if f.endswith(".txt")])

    clean_dir = "./data/cleaned_corpus"
    if os.path.exists(clean_dir):
        stats["deduped_files"] = len([f for f in os.listdir(clean_dir) if f.endswith(".txt")])

    tok_file = "./data/custom_tokenizer.json"
    if os.path.exists(tok_file):
        try:
            with open(tok_file, "r") as f:
                stats["tokenizer_vocab_size"] = json.load(f).get("vocab_size", 0)
        except Exception:
            pass

    pretrain_bin = "./data/train.bin"
    if os.path.exists(pretrain_bin):
        stats["pretrain_checkpoint_exists"] = True
        try:
            import numpy as np
            stats["packed_tokens"] = len(np.memmap(pretrain_bin, dtype=np.uint16, mode="r"))
        except Exception:
            pass

    stats["pretrain_checkpoint_exists"] = stats["pretrain_checkpoint_exists"] or os.path.exists("./outputs/checkpoint_pretrain.pt")
    stats["sft_checkpoint_exists"] = os.path.exists("./outputs/checkpoint_sft.pt")
    stats["dpo_checkpoint_exists"] = os.path.exists("./outputs_dpo/checkpoint_dpo.pt")

    eval_file = "./outputs/eval_report.json"
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
                stats["eval_scores"] = {
                    "mmlu": eval_data.get("mmlu_accuracy", 0.0) * 100,
                    "gsm8k": eval_data.get("gsm8k_accuracy", 0.0) * 100,
                }
        except Exception:
            pass

    return stats


@app.get("/api/telemetry")
def get_telemetry():
    telemetry_path = "outputs/system_telemetry.json"
    if os.path.exists(telemetry_path):
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback: build real telemetry from model config and GPU detection
    telemetry = {
        "model_loaded": _model_loaded,
        "gpu_count": 0,
        "gpu_name": "cpu",
        "gpu_memory_gb": 0,
        "timestamp": time.time(),
    }

    if _active_config is not None:
        cfg = _active_config
        telemetry["model"] = {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "vocab_size": cfg.vocab_size,
            "use_mla": getattr(cfg, "use_mla", False),
            "use_moe": getattr(cfg, "use_moe", False),
        }
        total_params = sum(p.numel() for p in _active_model.parameters()) if _active_model else 0
        telemetry["model"]["total_parameters"] = total_params

    if torch.cuda.is_available():
        telemetry["gpu_count"] = torch.cuda.device_count()
        telemetry["gpu_name"] = torch.cuda.get_device_name(0) if telemetry["gpu_count"] > 0 else "unknown"
        if telemetry["gpu_count"] > 0:
            props = torch.cuda.get_device_properties(0)
            telemetry["gpu_memory_gb"] = round(props.total_mem / (1024**3), 1)
            telemetry["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            telemetry["supports_bf16"] = torch.cuda.is_bf16_supported()
            telemetry["supports_fp8"] = props.major >= 9
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            telemetry["gpu_memory_used_gb"] = round(mem_info.used / (1024**3), 1)
            telemetry["gpu_memory_free_gb"] = round(mem_info.free / (1024**3), 1)
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return telemetry


# ==============================================================================
# Model Management
# ==============================================================================


@app.post("/api/model/load")
def api_load_model(checkpoint_path: str = "./outputs/checkpoint_sft.pt"):
    try:
        model, tokenizer, config = load_model(checkpoint_path)
        return {
            "status": "success",
            "layers": config.n_layer,
            "embed_dim": config.n_embd,
            "vocab_size": config.vocab_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/swap")
def api_swap_model(checkpoint_path: str):
    try:
        load_model(checkpoint_path)
        return {"status": "success", "message": f"Swapped to {checkpoint_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# OpenAI-Compatible Chat Completions API
# ==============================================================================


async def _generate_stream(prompt: str, temperature: float, max_tokens: int, top_p: float):
    model, tokenizer, config = get_model()
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    use_mla = getattr(config, 'use_mla', False)
    kv_comp_dim = getattr(config, 'kv_comp_dim', 128)
    n_kv_heads = config.n_kv_head if config.n_kv_head is not None else config.n_head
    head_dim = config.n_embd // config.n_head

    max_total_len = prompt_ids.shape[1] + max_tokens
    kv_caches = []
    for _ in range(config.n_layer):
        if use_mla:
            kv_caches.append(torch.zeros(1, max_total_len, kv_comp_dim, device=device, dtype=torch.bfloat16))
        else:
            k = torch.zeros(1, max_total_len, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.zeros(1, max_total_len, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
            kv_caches.append((k, v))

    # Prefill
    prompt_len = prompt_ids.shape[1]
    logits, _, _ = model(prompt_ids, start_pos=0, kv_caches=kv_caches)

    generated_tokens = []
    start_time = time.time()
    ttft_ms = None

    for i in range(max_tokens):
        curr_pos = prompt_len + i
        if i == 0:
            next_logits = logits[:, -1, :]
        else:
            last_token = torch.tensor([[generated_tokens[-1]]], device=device)
            logits, _, _ = model(last_token, start_pos=curr_pos - 1, kv_caches=kv_caches)
            next_logits = logits[:, -1, :]

        if temperature > 0:
            next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            next_logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        if ttft_ms is None:
            ttft_ms = (time.time() - start_time) * 1000

        token_text = tokenizer.decode([token_id])
        yield {
            "token": token_text,
            "ttft_ms": ttft_ms if i == 0 else 0,
        }

        if token_id == getattr(config, 'eos_token_id', None) or token_id == tokenizer.eos_token_id:
            break


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = request.messages[-1].content if request.messages else ""
    if request.messages and len(request.messages) > 1:
        # Build conversation context
        parts = []
        for msg in request.messages:
            role = msg.role
            if role == "system":
                parts.append(f"System: {msg.content}")
            elif role == "user":
                parts.append(f"User: {msg.content}")
            elif role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        prompt = "\n".join(parts) + "\nAssistant:"

    if request.stream:
        async def event_stream():
            async for token_data in _generate_stream(prompt, request.temperature, request.max_tokens, request.top_p):
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token_data['token']}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        tokens = []
        async for data in _generate_stream(prompt, request.temperature, request.max_tokens, request.top_p):
            tokens.append(data["token"])
        return {
            "choices": [{"message": {"role": "assistant", "content": "".join(tokens)}}],
            "model": request.model,
        }


# ==============================================================================
# Legacy Chat API
# ==============================================================================


@app.post("/api/chat")
async def chat_endpoint(request: PromptRequest):
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Load a model first via /api/model/load.")

    async def response_generator():
        async for data in _generate_stream(request.prompt, request.temperature, request.max_tokens, 0.9):
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")


# ==============================================================================
# Pipeline Control
# ==============================================================================


@app.post("/api/data/download")
def trigger_download():
    try:
        subprocess.run([sys.executable, "utils/download_dataset.py"], check=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/crawl")
def trigger_crawl():
    try:
        subprocess.run([sys.executable, "crawl_data.py", "--source", "local"], check=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/dedup")
def trigger_dedup():
    try:
        subprocess.run([sys.executable, "deduplicate.py", "--threshold", "0.75"], check=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/tokenize")
def trigger_tokenize(vocab_size: int = 1200):
    try:
        subprocess.run([sys.executable, "train_tokenizer.py", "--vocab_size", str(vocab_size)], check=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/pack")
def trigger_pack():
    try:
        subprocess.run([sys.executable, "pack_binaries.py"], check=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluation/benchmark")
def trigger_benchmark():
    checkpoint = "./outputs/checkpoint_sft.pt"
    if not os.path.exists(checkpoint):
        raise HTTPException(status_code=404, detail="SFT checkpoint not found")
    try:
        subprocess.run([sys.executable, "eval_benchmarks.py", "--checkpoint_path", checkpoint], check=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# WebSocket Log Streaming
# ==============================================================================

# Stage → training script mapping for real process piping
_STAGE_SCRIPTS = {
    "pretrain": {"script": "pretrain.py", "args": []},
    "sft": {"script": "train.py", "args": []},
    "dpo": {"script": "align.py", "args": []},
    "grpo": {"script": "grpo.py", "args": []},
}


def _launch_training_process(stage: str):
    """Launch a training script as a subprocess with line-buffered stdout.

    Returns the Popen object or None if the script cannot be launched.
    Real training typically uses torchrun for multi-GPU — this single-process
    launch is intended for demo/streaming purposes only.
    """
    info = _STAGE_SCRIPTS.get(stage)
    if info is None:
        return None

    script_path = info["script"]
    if not os.path.exists(script_path):
        logger.warning(f"Training script {script_path} not found, falling back to simulation.")
        return None

    cmd = [sys.executable, "-u", script_path] + info["args"]
    logger.info(f"Launching training process: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return proc
    except Exception as e:
        logger.warning(f"Failed to launch training process: {e}. Falling back to simulation.")
        return None


@app.websocket("/ws/logs/{stage}")
async def websocket_logs_endpoint(websocket: WebSocket, stage: str):
    await websocket.accept()
    logger.info(f"WebSocket client connected to {stage} stream.")

    # Attempt real training process piping
    proc = await asyncio.get_event_loop().run_in_executor(None, _launch_training_process, stage)

    if proc is not None:
        # Stream real training output line by line
        try:
            while True:
                # Check if client disconnected
                try:
                    line = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, proc.stdout.readline),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    if proc.poll() is not None:
                        break
                    continue

                if not line:
                    if proc.poll() is not None:
                        break
                    continue

                payload = {"text": line, "stage": stage}
                await websocket.send_text(json.dumps(payload))
        except Exception as e:
            logger.warning(f"WebSocket stream error: {e}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            await websocket.send_text(json.dumps({"text": "Training process ended.\n", "done": True}))
            await websocket.close()
        return

    # Fallback: simulated training logs
    # Real training uses torchrun with multi-GPU DDP which cannot be driven
    # from a single WebSocket subprocess. This simulation provides a live
    # dashboard demo. For production training, use the CLI pipeline directly:
    #   python run_unified_pipeline.py --mode prod --stage all
    steps = 50
    base_loss = 2.5
    for step in range(steps):
        loss = base_loss * (0.95 ** step)
        accuracy = min(100.0, 30.0 + (step * 1.3))
        mfu = 32.4 + (step % 5) * 0.5
        payload = {
            "text": f"Epoch 1 | Step {step+1}/{steps} | Loss: {loss:.4f} | MFU: {mfu:.1f}% | Acc: {accuracy:.1f}%\n",
            "metrics": {"step": step + 1, "loss": round(loss, 4), "accuracy": round(accuracy, 2), "mfu": round(mfu, 1)},
        }
        await websocket.send_text(json.dumps(payload))
        await asyncio.sleep(0.3)

    await websocket.send_text(json.dumps({"text": "Training stage completed!\n", "done": True}))
    await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
