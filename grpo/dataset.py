"""GRPO dataset and collate function."""

import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GRPODataset(Dataset):
    """Dataset of prompts, optional images, and expected answers for GRPO training."""

    def __init__(self, data_path: str, tokenizer, max_prompt_length: int = 1024, vision_dim: int = 1152, num_patches: int = 16):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.vision_dim = vision_dim
        self.num_patches = num_patches

        self.prompts = []
        self.images = []
        self.ground_truths = []

        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        self.prompts.append(data["prompt"])
                        self.images.append(data.get("image", None))
                        self.ground_truths.append(data.get("ground_truth", ""))
                    except Exception as e:
                        logger.warning(f"Error parsing line: {e}")
        else:
            logger.warning(f"Data path {data_path} not found. Loading mock mathematical reasoning dataset.")
            mock_prompts = [
                ("Solve: 12 + 15 * 2. Please reason step-by-step and wrap your final number in <answer></answer> tags.", None, "42"),
                ("What is the square of 9? Please think step-by-step and wrap your final number in <answer></answer> tags.", None, "81"),
                ("Compute: (25 - 5) / 4. Please show your thinking process and wrap your final answer in <answer></answer>.", None, "5"),
                ("What is 100 divided by 4? Reason and output final number inside <answer></answer> tags.", None, "25"),
            ]
            for p, img, gt in mock_prompts:
                self.prompts.append(p)
                self.images.append(img)
                self.ground_truths.append(gt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        image_path = self.images[idx]
        ground_truth = self.ground_truths[idx]

        inputs = self.tokenizer(
            prompt, max_length=self.max_prompt_length, truncation=True, return_tensors="pt"
        )

        if image_path is not None:
            from utils.vision_helper import extract_image_features
            pixel_values = extract_image_features(image_path, self.vision_dim, self.num_patches)
        else:
            pixel_values = None

        return {
            "prompt_text": prompt,
            "prompt_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": pixel_values,
            "ground_truth": ground_truth,
        }


def grpo_collate_fn(batch, pad_token_id=0):
    """Collate variable-length prompt token sequences and stack vision patches."""
    prompt_texts = [item["prompt_text"] for item in batch]
    prompt_ids_list = [item["prompt_ids"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]

    padded_prompt_ids = nn.utils.rnn.pad_sequence(prompt_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_masks = (padded_prompt_ids != pad_token_id).long()

    pixel_values_list = []
    has_images = False
    for item in batch:
        p_val = item["pixel_values"]
        if p_val is not None:
            has_images = True
            pixel_values_list.append(p_val)
        else:
            pixel_values_list.append(torch.zeros(16, 1152))

    batch_pixel_values = torch.stack(pixel_values_list) if has_images else None

    return {
        "prompt_texts": prompt_texts,
        "prompt_ids": padded_prompt_ids,
        "attention_masks": attention_masks,
        "pixel_values": batch_pixel_values,
        "ground_truths": ground_truths,
    }
