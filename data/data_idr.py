import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import torch
import yaml
from utils.utils import load_configs
import json

class IDRDataset(Dataset):
    def __init__(self, data, max_len=1022):
        """
        data: List of dicts with 'id', 'sequence', 'reference'
        reference values: '0', '1', or '-'
        """
        self.data = data
        self.max_len = max_len # ESM-2 standard limit

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['sequence'][:self.max_len] # Truncate sequence
        ref_str = item['reference'][:self.max_len] # Truncate labels
        
        label_map = {'0': 0, '1': 1, '-': -1}
        labels = torch.tensor([label_map.get(c, -1) for c in ref_str], dtype=torch.long)
        
        return {"id": item['id'], "seq": seq, "labels": labels}


def parse_fasta_idr(path):
    """Parses your specific FASTA format with reference lines below sequences."""
    entries = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 3):
            header = lines[i].replace('>', '')
            seq = lines[i+1]
            ref = lines[i+2]
            entries.append({"id": header, "sequence": seq, "reference": ref})
    return entries

def collate_idr(batch):
    return [b['id'] for b in batch], [b['seq'] for b in batch], [b['labels'] for b in batch]

def prepare_dataloaders_idr(configs):
    json_path = configs.train_settings.data_path
    test_fasta_path = configs.test_settings.data_path
    with open(json_path, 'r') as f:
        train_data = json.load(f)
    
    train_idx, val_idx = train_test_split(range(len(train_data)), test_size=0.2, random_state=42)
    train_set = IDRDataset([train_data[i] for i in train_idx])
    val_set = IDRDataset([train_data[i] for i in val_idx])
    
    test_data = parse_fasta_idr(test_fasta_path)
    test_set = IDRDataset(test_data)

    return {
        'train': DataLoader(train_set, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=collate_idr),
        'valid': DataLoader(val_set, batch_size=configs.valid_settings.batch_size, collate_fn=collate_idr),
        'test': DataLoader(test_set, batch_size=configs.test_settings.batch_size, collate_fn=collate_idr)
    }

if __name__ == '__main__':
    print('no test')
    # config_path = './config_enzyme_reaction.yaml'
    # with open(config_path) as file:
    #     configs_dict = yaml.full_load(file)

    # configs_file = load_configs(configs_dict)

    # dataloaders_dict = prepare_dataloaders_fold(configs_file)
    # max_position_value = []
    # amino_acid = []
    # for batch in dataloaders_dict['train']:
    #     sequence_batch, label_batch, position_batch, weights_batch = batch
    #     # print(sequence_batch['input_ids'].shape)
    #     # print(label_batch.shape)
    #     # print(position_batch.shape)
    #     max_position_value.append(position_batch.squeeze().numpy().item())
    #     amino_acid.append(sequence_batch["input_ids"][0][position_batch.squeeze().numpy().item()].item())
    # print(set(max_position_value))
    # print([dataloaders_dict['train'].dataset.encoder_tokenizer.id_to_token(i) for i in set(amino_acid)])
    # print('done')