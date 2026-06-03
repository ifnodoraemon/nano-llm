import json
import logging
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ChatFormatter:
    """
    Handles formatting of raw multi-turn conversation lists into standard ChatML text.
    """
    def __init__(self):
        self.system_template = "<|im_start|>system\n{content}<|im_end|>\n"
        self.user_template = "<|im_start|>user\n{content}<|im_end|>\n"
        self.assistant_template = "<|im_start|>assistant\n{content}<|im_end|>\n"

    def format_message(self, message: Dict[str, str]) -> str:
        role = message["role"]
        content = message["content"]
        if role == "system":
            return self.system_template.format(content=content)
        elif role == "user":
            return self.user_template.format(content=content)
        elif role == "assistant":
            return self.assistant_template.format(content=content)
        else:
            raise ValueError(f"Unsupported message role: {role}")


class SFTDataset(Dataset):
    """
    PyTorch Dataset that loads conversational dialogues in JSON Lines format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatter = ChatFormatter()
        
        logger.info(f"Loading conversational data from {data_path}...")
        self.dialogues = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "messages" in data:
                        self.dialogues.append(data["messages"])
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    
        logger.info(f"Loaded {len(self.dialogues)} conversations.")

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        messages = self.dialogues[idx]
        
        input_ids = []
        labels = []
        
        for msg in messages:
            role = msg["role"]
            formatted_text = self.formatter.format_message(msg)
            try:
                tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            except TypeError:
                tokens = self.tokenizer.encode(formatted_text)
            
            input_ids.extend(tokens)
            
            # Label masking: Only compute loss on assistant responses
            # User prompts/system prompts are labeled with -100 to ignore in cross-entropy loss
            if role == "assistant":
                labels.extend(tokens)
            else:
                labels.extend([-100] * len(tokens))
                
        # Truncate if sequence exceeds max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class SequencePackingCollator:
    """
    Fuses multiple variable-length conversation samples into single packed sequences 
    of length `max_length` to prevent waste on pad tokens.
    Generates customized position_ids that reset at sequence boundaries.
    """
    def __init__(self, pad_token_id: int, max_length: int = 4096, batch_size: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        batch_seqlens = []
        
        curr_input_ids = []
        curr_labels = []
        curr_position_ids = []
        curr_seqlens = []
        
        for sample in samples:
            ids = sample["input_ids"].tolist()
            lbls = sample["labels"].tolist()
            seq_len = len(ids)
            
            # Handle exceptionally long sequences
            if seq_len > self.max_length:
                ids = ids[:self.max_length]
                lbls = lbls[:self.max_length]
                seq_len = self.max_length
                
            # If adding the sample exceeds max_length, pack current buffer and start a new one
            if len(curr_input_ids) + seq_len > self.max_length:
                padding_len = self.max_length - len(curr_input_ids)
                if padding_len > 0:
                    curr_input_ids.extend([self.pad_token_id] * padding_len)
                    curr_labels.extend([-100] * padding_len)
                    curr_position_ids.extend([0] * padding_len)
                    curr_seqlens.append(padding_len)
                    
                batch_input_ids.append(torch.tensor(curr_input_ids, dtype=torch.long))
                batch_labels.append(torch.tensor(curr_labels, dtype=torch.long))
                batch_position_ids.append(torch.tensor(curr_position_ids, dtype=torch.long))
                batch_seqlens.append(curr_seqlens)
                
                # Reset buffers
                curr_input_ids = []
                curr_labels = []
                curr_position_ids = []
                curr_seqlens = []
                
            curr_input_ids.extend(ids)
            curr_labels.extend(lbls)
            curr_position_ids.extend(list(range(seq_len))) # Reset positions at boundaries!
            curr_seqlens.append(seq_len)
            
        # Append remaining buffer
        if curr_input_ids:
            padding_len = self.max_length - len(curr_input_ids)
            if padding_len > 0:
                curr_input_ids.extend([self.pad_token_id] * padding_len)
                curr_labels.extend([-100] * padding_len)
                curr_position_ids.extend([0] * padding_len)
                curr_seqlens.append(padding_len)
                
            batch_input_ids.append(torch.tensor(curr_input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(curr_labels, dtype=torch.long))
            batch_position_ids.append(torch.tensor(curr_position_ids, dtype=torch.long))
            batch_seqlens.append(curr_seqlens)
            
        # Static shape padding or truncation to prevent torch.compile recompilations
        if self.batch_size is not None:
            if len(batch_input_ids) > self.batch_size:
                batch_input_ids = batch_input_ids[:self.batch_size]
                batch_labels = batch_labels[:self.batch_size]
                batch_position_ids = batch_position_ids[:self.batch_size]
                batch_seqlens = batch_seqlens[:self.batch_size]
            elif len(batch_input_ids) < self.batch_size:
                needed = self.batch_size - len(batch_input_ids)
                for _ in range(needed):
                    batch_input_ids.append(torch.full((self.max_length,), self.pad_token_id, dtype=torch.long))
                    batch_labels.append(torch.full((self.max_length,), -100, dtype=torch.long))
                    batch_position_ids.append(torch.zeros(self.max_length, dtype=torch.long))
                    batch_seqlens.append([self.max_length])

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.stack(batch_position_ids),
            "seqlens": batch_seqlens,
            "attention_mask": torch.ones(len(batch_input_ids), self.max_length, dtype=torch.long)
        }


class DPODataset(Dataset):
    """
    Dataset that loads pairwise preference samples for DPO training:
    {
      "prompt": [{"role": "user", "content": "..."}, ...],
      "chosen": {"role": "assistant", "content": "..."},
      "rejected": {"role": "assistant", "content": "..."}
    }
    """
    def __init__(self, data_path: str, tokenizer, max_prompt_length: int = 2048, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length
        self.formatter = ChatFormatter()
        
        logger.info(f"Loading preference pairs from {data_path}...")
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if all(k in data for k in ["prompt", "chosen", "rejected"]):
                        self.samples.append(data)
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    
        logger.info(f"Loaded {len(self.samples)} preference alignment samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        
        # 1. Format and tokenize prompt context
        prompt_text = ""
        for msg in item["prompt"]:
            prompt_text += self.formatter.format_message(msg)
            
        try:
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        except TypeError:
            prompt_ids = self.tokenizer.encode(prompt_text)
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:] # Keep latest turns
            
        # 2. Format and tokenize chosen & rejected answers
        chosen_text = self.formatter.format_message(item["chosen"])
        rejected_text = self.formatter.format_message(item["rejected"])
        
        try:
            chosen_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False)
        except TypeError:
            chosen_ids = self.tokenizer.encode(chosen_text)
        try:
            rejected_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)
        except TypeError:
            rejected_ids = self.tokenizer.encode(rejected_text)
        
        # Combine prompt with chosen & rejected up to max_length
        max_ans_len = self.max_length - len(prompt_ids)
        chosen_ids = chosen_ids[:max_ans_len]
        rejected_ids = rejected_ids[:max_ans_len]
        
        # Input ids for policy evaluation
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        
        # Labels: Mask prompt tokens out (value = -100)
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids
        
        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long)
        }


# ==============================================================================
# Native Multimodal Vision-Language Dataset & Collator (VLM Supported)
# ==============================================================================

class MultimodalSFTDataset(Dataset):
    """
    Loads conversational dialogues in JSON Lines format with an optional image key:
    {
      "messages": [{"role": "user", "content": "<image> Describe this image."}, ...],
      "image": "path/to/image.jpg" # Optional local path or URL
    }
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096, vision_dim: int = 1152, num_patches: int = 16):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vision_dim = vision_dim
        self.num_patches = num_patches
        self.formatter = ChatFormatter()
        
        logger.info(f"Loading multimodal SFT data from {data_path}...")
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "messages" in data:
                        self.samples.append(data)
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    
        logger.info(f"Loaded {len(self.samples)} multimodal conversations.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        messages = item["messages"]
        image_path = item.get("image", None)
        
        input_ids = []
        labels = []
        
        for msg in messages:
            role = msg["role"]
            formatted_text = self.formatter.format_message(msg)
            try:
                tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            except TypeError:
                tokens = self.tokenizer.encode(formatted_text)
            
            input_ids.extend(tokens)
            
            if role == "assistant":
                labels.extend(tokens)
            else:
                labels.extend([-100] * len(tokens))
                
        # Truncate if sequence exceeds max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        # Process image features if image exists
        if image_path is not None and os.path.exists(image_path):
            try:
                from PIL import Image
                
                img = Image.open(image_path).convert("RGB")
                
                # Check if we can load SigLIP processor and model to extract real features
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                from transformers import AutoProcessor, AutoModel
                
                # Compact SigLIP model to extract real features
                model_name = "google/siglip-base-patch16-224"
                processor = AutoProcessor.from_pretrained(model_name, local_files_only=False)
                vision_model = AutoModel.from_pretrained(model_name, local_files_only=False)
                
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = vision_model.vision_model(inputs.pixel_values)
                    # Shape: [num_patches, vision_dim] (e.g. 196 patches, 768 dim)
                    last_hidden_state = outputs.last_hidden_state.squeeze(0)
                    
                # If number of patches or vision dim does not match, project it
                if last_hidden_state.shape[-1] != self.vision_dim:
                    proj = torch.nn.Linear(last_hidden_state.shape[-1], self.vision_dim)
                    pixel_values = proj(last_hidden_state).detach()
                else:
                    pixel_values = last_hidden_state
            except Exception as e:
                logger.warning(f"Could not extract features using SigLIP model ({e}). Falling back to simple patch extraction.")
                try:
                    img = Image.open(image_path).convert("RGB")
                    img = img.resize((224, 224))
                    img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                    
                    patch_w = img_t.shape[1] // int(math.sqrt(self.num_patches))
                    patches = []
                    for r in range(int(math.sqrt(self.num_patches))):
                        for c in range(int(math.sqrt(self.num_patches))):
                            patch = img_t[r*patch_w:(r+1)*patch_w, c*patch_w:(c+1)*patch_w, :]
                            patches.append(patch.flatten())
                    flat_patches = torch.stack(patches)
                    proj = torch.nn.Linear(flat_patches.shape[-1], self.vision_dim)
                    pixel_values = proj(flat_patches).detach()
                except Exception as ex:
                    logger.error(f"Image load fallback failed: {ex}. Falling back to random projection.")
                    pixel_values = torch.randn(self.num_patches, self.vision_dim)
        else:
            if image_path is not None:
                logger.warning(f"Multimodal image path not found: {image_path}. Using random features.")
                pixel_values = torch.randn(self.num_patches, self.vision_dim)
            else:
                pixel_values = None
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values
        }


class MultimodalSequenceCollator:
    """
    Collate function that pads multimodal batches, stacking token lists, targets,
    and visual features into consistent tensors.
    """
    def __init__(self, pad_token_id: int, vision_dim: int = 1152, num_patches: int = 16):
        self.pad_token_id = pad_token_id
        self.vision_dim = vision_dim
        self.num_patches = num_patches

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad tokens to matching length (longest sequence in batch)
        max_seq_len = max(len(sample["input_ids"]) for sample in samples)
        
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        has_images = False
        
        for sample in samples:
            ids = sample["input_ids"]
            lbls = sample["labels"]
            padding_len = max_seq_len - len(ids)
            
            if padding_len > 0:
                padded_ids = torch.cat([ids, torch.full((padding_len,), self.pad_token_id, dtype=torch.long)])
                padded_lbls = torch.cat([lbls, torch.full((padding_len,), -100, dtype=torch.long)])
            else:
                padded_ids = ids
                padded_lbls = lbls
                
            batch_input_ids.append(padded_ids)
            batch_labels.append(padded_lbls)
            
            # Handle pixel values
            p_val = sample["pixel_values"]
            if p_val is not None:
                has_images = True
                batch_pixel_values.append(p_val)
            else:
                # Pad missing image features with zero maps
                batch_pixel_values.append(torch.zeros(self.num_patches, self.vision_dim))
                
        collated = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack([
                torch.cat([torch.ones(len(s["input_ids"]), dtype=torch.long), torch.zeros(max_seq_len - len(s["input_ids"]), dtype=torch.long)])
                for s in samples
            ])
        }
        
        if has_images:
            collated["pixel_values"] = torch.stack(batch_pixel_values)
            
        return collated


class MultimodalDPODataset(Dataset):
    """
    Multimodal DPO Dataset containing prompt (with optional image) + chosen + rejected responses.
    """
    def __init__(self, data_path: str, tokenizer, max_prompt_length: int = 2048, max_length: int = 4096, vision_dim: int = 1152, num_patches: int = 16):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length
        self.vision_dim = vision_dim
        self.num_patches = num_patches
        self.formatter = ChatFormatter()
        
        logger.info(f"Loading multimodal DPO preference pairs from {data_path}...")
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if all(k in data for k in ["prompt", "chosen", "rejected"]):
                        self.samples.append(data)
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    
        logger.info(f"Loaded {len(self.samples)} multimodal preference alignment samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_path = item.get("image", None)
        
        # 1. Format and tokenize prompt context
        prompt_text = ""
        for msg in item["prompt"]:
            prompt_text += self.formatter.format_message(msg)
            
        try:
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        except TypeError:
            prompt_ids = self.tokenizer.encode(prompt_text)
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:]
            
        # 2. Format and tokenize chosen & rejected answers
        chosen_text = self.formatter.format_message(item["chosen"])
        rejected_text = self.formatter.format_message(item["rejected"])
        
        try:
            chosen_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False)
        except TypeError:
            chosen_ids = self.tokenizer.encode(chosen_text)
        try:
            rejected_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)
        except TypeError:
            rejected_ids = self.tokenizer.encode(rejected_text)
            
        # Combine prompt with chosen & rejected up to max_length
        max_ans_len = self.max_length - len(prompt_ids)
        chosen_ids = chosen_ids[:max_ans_len]
        rejected_ids = rejected_ids[:max_ans_len]
        
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids
        
        # Extract image features if present
        pixel_values = None
        if image_path is not None and os.path.exists(image_path):
            try:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                from transformers import AutoProcessor, AutoModel
                model_name = "google/siglip-base-patch16-224"
                processor = AutoProcessor.from_pretrained(model_name, local_files_only=False)
                vision_model = AutoModel.from_pretrained(model_name, local_files_only=False)
                
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = vision_model.vision_model(inputs.pixel_values)
                    last_hidden_state = outputs.last_hidden_state.squeeze(0)
                    
                if last_hidden_state.shape[-1] != self.vision_dim:
                    proj = torch.nn.Linear(last_hidden_state.shape[-1], self.vision_dim)
                    pixel_values = proj(last_hidden_state).detach()
                else:
                    pixel_values = last_hidden_state
            except Exception as e:
                logger.warning(f"Could not extract features using SigLIP model ({e}). Falling back to simple patch extraction.")
                try:
                    img = Image.open(image_path).convert("RGB")
                    img = img.resize((224, 224))
                    img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                    patch_w = img_t.shape[1] // int(math.sqrt(self.num_patches))
                    patches = []
                    for r in range(int(math.sqrt(self.num_patches))):
                        for c in range(int(math.sqrt(self.num_patches))):
                            patch = img_t[r*patch_w:(r+1)*patch_w, c*patch_w:(c+1)*patch_w, :]
                            patches.append(patch.flatten())
                    flat_patches = torch.stack(patches)
                    proj = torch.nn.Linear(flat_patches.shape[-1], self.vision_dim)
                    pixel_values = proj(flat_patches).detach()
                except Exception as ex:
                    logger.error(f"Image load fallback failed: {ex}. Falling back to random projection.")
                    pixel_values = torch.randn(self.num_patches, self.vision_dim)
        else:
            if image_path is not None:
                logger.warning(f"Multimodal image path not found: {image_path}. Using random features.")
                pixel_values = torch.randn(self.num_patches, self.vision_dim)
                
        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "pixel_values": pixel_values
        }

