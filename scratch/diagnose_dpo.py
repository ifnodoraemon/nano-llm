import torch
import torch.nn as nn
from model import Transformer
from data import DPODataset
from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
from utils.tokenizer_loader import load_tokenizer
from utils.ddp_helper import init_ddp
from torch.utils.data import DataLoader, DistributedSampler

def main():
    dist_info = init_ddp()
    ddp_rank = dist_info['rank']
    device = dist_info['device']
    ddp_world_size = dist_info['world_size']

    tokenizer = load_tokenizer()
    dataset = DPODataset(data_path='./data/train_dpo_premium.jsonl', tokenizer=tokenizer)

    def dpo_collator(samples):
        batch = {}
        for key in ['chosen_input_ids', 'chosen_labels', 'rejected_input_ids', 'rejected_labels']:
            tensor_list = [item[key] for item in samples]
            max_len = max(x.size(0) for x in tensor_list)
            pad_val = tokenizer.pad_token_id if 'input_ids' in key else -100
            batch[key] = torch.stack([
                torch.cat([x, torch.tensor([pad_val] * (max_len - x.size(0)), dtype=torch.long)])
                for x in tensor_list
            ])
        return batch

    sampler = DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=dpo_collator)

    model_config, state_dict = load_checkpoint_with_fp8_translation('./outputs_dev/checkpoint_sft.pt', map_location='cpu')
    model = Transformer(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f'Rank {ddp_rank} initialized. Checking all batches...')

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for prefix in ['chosen', 'rejected']:
                input_ids = batch[f'{prefix}_input_ids'].to(device)
                labels = batch[f'{prefix}_labels'].to(device)
                
                # Forward pass
                logits, _, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Check bounds
                vocab_size = shift_logits.size(-1)
                invalid_mask = (shift_labels >= vocab_size) | ((shift_labels < 0) & (shift_labels != -100))
                if invalid_mask.any():
                    print(f'❌ Rank {ddp_rank} | Batch {batch_idx} | Prefix {prefix} | Invalid labels detected!')
                    print(f'Logits vocab size: {vocab_size}')
                    invalid_vals = shift_labels[invalid_mask].tolist()
                    print('Invalid labels:', invalid_vals)
                    print('Invalid input ids at same positions:', input_ids[..., 1:][invalid_mask].tolist())

if __name__ == "__main__":
    main()
