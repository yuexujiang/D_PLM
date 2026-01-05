import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import torch
import yaml
from utils.utils import load_configs

class StabilityDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "protein_id": row["protein_id"],
            "wt_seq": row["wt_seq"],
            "mut_seq": row["mut_seq"],
            "ddg": torch.tensor(row["ddG"], dtype=torch.float32)
        }
    
def extract_protein_id(name):
    """Extract protein ID from name: rcsb_1A7V_A_A104H_6_25 ? 1A7V_A"""
    parts = name.split("_")
    return f"{parts[1]}_{parts[2]}"

def collate_fn(batch):
    """Collate function for DataLoader"""
    return (
        [b["wt_seq"] for b in batch],
        [b["mut_seq"] for b in batch],
        torch.stack([b["ddg"] for b in batch]),
        [b["protein_id"] for b in batch],
    )

def prepare_dataloaders_fold(configs):
    df = pd.read_csv("./evaluate/ddt/S8754.csv")
    df["protein_id"] = df["name"].apply(extract_protein_id)
    proteins = df["protein_id"].unique()
    train_prots, val_prots = train_test_split(
        proteins, test_size=0.2, random_state=42
    )
    train_df = df[df["protein_id"].isin(train_prots)]
    val_df = df[df["protein_id"].isin(val_prots)]
    train_dataset = StabilityDataset(train_df)
    val_dataset = StabilityDataset(val_df)

    df_test = pd.read_csv("./evaluate/ddt/S669.csv")
    df_test["protein_id"] = df_test["name"].apply(extract_protein_id)
    test_dataset = StabilityDataset(df_test)

    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                          shuffle=True, collate_fn=collate_fn)
    val_dataloader   = DataLoader(val_dataset, batch_size=configs.valid_settings.batch_size,
                          shuffle=False, collate_fn=collate_fn)

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=configs.test_settings.batch_size,
        collate_fn=collate_fn
    )
    return {'train': train_dataloader, 'valid': val_dataloader, 
            'test': test_dataloader}

if __name__ == '__main__':
    config_path = './config_enzyme_reaction.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dataloaders_dict = prepare_dataloaders_fold(configs_file)
    max_position_value = []
    amino_acid = []
    for batch in dataloaders_dict['train']:
        sequence_batch, label_batch, position_batch, weights_batch = batch
        # print(sequence_batch['input_ids'].shape)
        # print(label_batch.shape)
        # print(position_batch.shape)
        max_position_value.append(position_batch.squeeze().numpy().item())
        amino_acid.append(sequence_batch["input_ids"][0][position_batch.squeeze().numpy().item()].item())
    print(set(max_position_value))
    print([dataloaders_dict['train'].dataset.encoder_tokenizer.id_to_token(i) for i in set(amino_acid)])
    print('done')