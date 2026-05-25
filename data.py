import json
import logging
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
            tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            
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
    def __init__(self, pad_token_id: int, max_length: int = 4096):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        
        curr_input_ids = []
        curr_labels = []
        curr_position_ids = []
        
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
                    
                batch_input_ids.append(torch.tensor(curr_input_ids, dtype=torch.long))
                batch_labels.append(torch.tensor(curr_labels, dtype=torch.long))
                batch_position_ids.append(torch.tensor(curr_position_ids, dtype=torch.long))
                
                # Reset buffers
                curr_input_ids = []
                curr_labels = []
                curr_position_ids = []
                
            curr_input_ids.extend(ids)
            curr_labels.extend(lbls)
            curr_position_ids.extend(list(range(seq_len))) # Reset positions at boundaries!
            
        # Append remaining buffer
        if curr_input_ids:
            padding_len = self.max_length - len(curr_input_ids)
            if padding_len > 0:
                curr_input_ids.extend([self.pad_token_id] * padding_len)
                curr_labels.extend([-100] * padding_len)
                curr_position_ids.extend([0] * padding_len)
                
            batch_input_ids.append(torch.tensor(curr_input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(curr_labels, dtype=torch.long))
            batch_position_ids.append(torch.tensor(curr_position_ids, dtype=torch.long))
            
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.stack(batch_position_ids),
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
            
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:] # Keep latest turns
            
        # 2. Format and tokenize chosen & rejected answers
        chosen_text = self.formatter.format_message(item["chosen"])
        rejected_text = self.formatter.format_message(item["rejected"])
        
        chosen_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)
        
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
