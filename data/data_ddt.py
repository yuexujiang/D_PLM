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

def apply_mutations_0based(wt_seq: str, mut_info: str) -> str:
    """
    Apply multiple mutations like:
      "A140C,C156S,I159V,F162I"
    Positions are 0-based.
    """
    seq = list(wt_seq)

    if isinstance(mut_info, float) and pd.isna(mut_info):
        # no mutation
        return wt_seq

    mutations = [m.strip() for m in mut_info.split(",") if m.strip()]

    for m in mutations:
        if len(m) < 3:
            raise ValueError(f"Bad mutation token: {m}")

        wt_aa = m[0]
        mut_aa = m[-1]
        pos_str = m[1:-1]

        if not pos_str.isdigit():
            raise ValueError(f"Bad mutation position: {m}")

        pos = int(pos_str)  # 0-based index

        if pos < 0 or pos >= len(seq):
            raise ValueError(
                f"Mutation position out of range: {m} (len={len(seq)})"
            )

        # sanity check (strongly recommended for DMS)
        if seq[pos] != wt_aa:
            raise ValueError(
                f"WT mismatch for mutation '{m}': "
                f"sequence has '{seq[pos]}' at position {pos}, expected '{wt_aa}'"
            )

        seq[pos] = mut_aa

    return "".join(seq)

def read_fasta_to_dict(fasta_path: str) -> dict:
    """Parse FASTA into dict: {header: sequence} (header is first token after '>')."""
    seqs = {}
    cur_name = None
    cur = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_name is not None:
                    seqs[cur_name] = "".join(cur)
                cur_name = line[1:].split()[0]  # take first token
                cur = []
            else:
                cur.append(line)
    if cur_name is not None:
        seqs[cur_name] = "".join(cur)
    return seqs

def prepare_dataloaders_from_dms(configs):
    """
    Build train/valid dataloaders from DMS CSV + WT fasta.
    Keeps StabilityDataset, uses CSV 'label' as ddG/stability score.
    """
    # 1) load WT sequences
    wt_dict = read_fasta_to_dict("./evaluate/ddt/dms_wt.fasta")

    # 2) load DMS
    df = pd.read_csv("./evaluate/ddt/dms.csv")
    # expected columns: protein_name, mut_info, label

    # 3) build wt_seq + mut_seq + ddG + protein_id
    def _get_wt(name):
        if name not in wt_dict:
            raise KeyError(f"WT fasta does not contain protein_name='{name}'")
        return wt_dict[name]

    df["protein_id"] = df["protein_name"]
    df["wt_seq"] = df["protein_name"].apply(_get_wt)

    # apply mutation to create mut_seq
    df["mut_seq"] = [
        apply_mutations_0based(wt, mut)
        for wt, mut in zip(df["wt_seq"].tolist(), df["mut_info"].tolist())
    ]

    # rename label -> ddG (stability score)
    df["ddG"] = df["label"].astype(float)

    # 4) split train/valid (protein-wise split to avoid leakage)
    proteins = df["protein_id"].unique()
    train_prots, val_prots = train_test_split(proteins, test_size=0.2, random_state=42)

    train_df = df[df["protein_id"].isin(train_prots)].reset_index(drop=True)
    val_df   = df[df["protein_id"].isin(val_prots)].reset_index(drop=True)

    train_dataset = StabilityDataset(train_df)
    val_dataset   = StabilityDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs.valid_settings.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return {"train": train_loader, "valid": val_loader}

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