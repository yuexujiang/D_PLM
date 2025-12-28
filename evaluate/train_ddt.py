import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import argparse
import torch
import esm
import esm_adapterH
import yaml
from utils.utils import load_configs
from collections import OrderedDict
from peft import LoraConfig, get_peft_model


import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

import os

import torch.nn as nn
import torch
import esm
import esm_adapterH
import torch.backends.cudnn as cudnn
import numpy as np
# import pylab
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import math
from collections import OrderedDict
from inspect import signature
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from Bio.PDB import PDBParser, PPBuilder
# ----------------------------
# Setup
# ----------------------------
parser = argparse.ArgumentParser(description="DMS regression tasks")
parser.add_argument('--model_type', type=str, required=False, default="s-plm")
parser.add_argument('--config_path', type=str, required=False, default="../results/vivit_2/config.yaml")
parser.add_argument('--model_location', type=str, required=False, 
                    default="../results/vivit_2/checkpoints/checkpoint_best_val_dms_corr.pth")
parser.add_argument('--loss', type=str, required=False, default="mse")


args = parser.parse_args(args=['--loss', 'mse',
                               '--model_type', "s-plm", #"s-plm" "ESM2"
                               '--config_path', "/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_3/config_vivit3.yaml",
                               '--model_location', "/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth"]
                               )




def load_checkpoints(model,checkpoint_path):
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            if  any('adapter' in name for name, _ in model.named_modules()):
                if np.sum(["adapter_layer_dict" in key for key in checkpoint[
                    'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
                    new_ordered_dict = OrderedDict()
                    for key, value in checkpoint['state_dict1'].items():
                        if "adapter_layer_dict" not in key:
                            new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                            new_ordered_dict[new_key] = value
                        else:
                            new_ordered_dict[key] = value
                    
                    model.load_state_dict(new_ordered_dict, strict=False)
                else:
                    #this model does not contain esm2
                    new_ordered_dict = OrderedDict()
                    for key, value in checkpoint['state_dict1'].items():
                            key = key.replace("esm2.","")
                            new_ordered_dict[key] = value
                    
                    model.load_state_dict(new_ordered_dict, strict=False)
            elif  any('lora' in name for name, _ in model.named_modules()):
                #this model does not contain esm2
                new_ordered_dict = OrderedDict()
                for key, value in checkpoint['state_dict1'].items():
                        print(key)
                        key = key.replace("esm2.","")
                        new_ordered_dict[key] = value
                
                model.load_state_dict(new_ordered_dict, strict=False)
            
            print("checkpoints were loaded from " + checkpoint_path)
        else:
            print("checkpoints not exist "+ checkpoint_path)

def load_model(args):
    if args.model_type=="s-plm":
        with open(args.config_path) as file:
            config_file = yaml.full_load(file)
            configs = load_configs(config_file, args=None)
        if configs.model.esm_encoder.adapter_h.enable:
            model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)
        elif configs.model.esm_encoder.lora.enable:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            lora_targets =  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj","self_attn.out_proj"]
            target_modules=[]
            if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                start_layer_idx = np.max([model.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                for idx in range(start_layer_idx, model.num_layers):
                    for layer_name in lora_targets:
                        target_modules.append(f"layers.{idx}.{layer_name}")
                
            peft_config = LoraConfig(
                inference_mode=False,
                r=configs.model.esm_encoder.lora.r,
                lora_alpha=configs.model.esm_encoder.lora.alpha,
                target_modules=target_modules,
                lora_dropout=configs.model.esm_encoder.lora.dropout,
                bias="none",
                # modules_to_save=modules_to_save
            )
            peft_model = get_peft_model(model, peft_config)
        
        # inference for each model
        # model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)/
        load_checkpoints(model,args.model_location)
    elif args.model_type=="ESM2":
        #if use ESM2
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    
    return model,alphabet

args.model, args.alphabet = load_model(args)
# batch_converter = args.alphabet.get_batch_converter(truncation_seq_length=512)
batch_converter = args.alphabet.get_batch_converter()




log_filename = f'training_ddt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Still print to console
    ]
)
logger = logging.getLogger(__name__)

def extract_protein_id(name):
    """Extract protein ID from name: rcsb_1A7V_A_A104H_6_25 ? 1A7V_A"""
    parts = name.split("_")
    return f"{parts[1]}_{parts[2]}"

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

class MLPHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def create_encoder_function(model, batch_converter):
    """
    Wrapper to create encoder function with your pretrained model.
    Assumes model and batch_converter are already loaded.
    """
    def encoder(seqs):
        batch_seq = [("seq_" + str(i), str(seqs[i])) for i in range(len(seqs))]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
        
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[model.num_layers])
            mt_representation = results["representations"][model.num_layers]
        
        # Average over sequence length: [B, L, dim] -> [B, dim]
        rep = torch.mean(mt_representation, axis=1)
        return rep
    
    return encoder

class StabilityModel(nn.Module):
    def __init__(self, encoder, embed_dim, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.mlp = MLPHead(embed_dim)
        self.freeze_encoder = freeze_encoder

    def encode(self, seqs):
        """Encode sequences using the pretrained encoder"""
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(seqs)
        else:
            return self.encoder(seqs)

    def forward(self, wt_seqs, mut_seqs):
        """
        Forward pass: encode both WT and mutant sequences,
        predict their scores, and return the difference (ddG prediction)
        """
        wt_emb = self.encode(wt_seqs)
        mut_emb = self.encode(mut_seqs)

        wt_score = self.mlp(wt_emb)
        mut_score = self.mlp(mut_emb)

        # Return difference: mut - wt
        return (mut_score - wt_score).squeeze(-1)

def soft_rank(x, tau=1.0):
    """Compute soft ranking using sigmoid approximation"""
    x = x.unsqueeze(0)  # (1, N)
    diff = x.T - x  # (N, N)
    P = torch.sigmoid(diff / tau)
    return P.sum(dim=1) + 1

def soft_spearman(pred, target, tau=1.0):
    """Compute differentiable Spearman correlation"""
    if pred.std() < 1e-6 or target.std() < 1e-6:
        # Return 0 correlation if no variance (will result in loss of 1.0, which is ignored)
        return torch.tensor(0.0, device=pred.device)
    
    pr = soft_rank(pred, tau)
    tr = soft_rank(target, tau)

    pr = pr - pr.mean()
    tr = tr - tr.mean()

    numerator = (pr * tr).sum()
    pr_norm = torch.sqrt((pr ** 2).sum())
    tr_norm = torch.sqrt((tr ** 2).sum())
    
    # Check for zero norms
    if pr_norm < 1e-6 or tr_norm < 1e-6:
        return torch.tensor(0.0, device=pred.device)
    
    denominator = pr_norm * tr_norm + 1e-8
    
    # Clamp to avoid numerical issues
    corr = numerator / denominator
    corr = torch.clamp(corr, -1.0, 1.0)
    return corr

def soft_spearman_loss(pred, target, tau=1.0):
    """Loss version: 1 - correlation"""
    corr = soft_spearman(pred, target, tau)
    # If correlation is 0 (invalid), return 0 loss so it doesn't contribute
    if torch.abs(corr) < 1e-6:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - corr

def train_step(model, batch, optimizer, lambda_rank=0.5, tau=0.5, chunk_size=32, loss_type='mse'):
    """
    Training step with combined MSE and soft Spearman loss.
    Soft Spearman is computed per protein (only for proteins with 2+ mutations).
    """
    model.train()
    optimizer.zero_grad()

    wt, mut, ddg, prot_ids = batch
    # pred = model(wt, mut)
    batch_size = len(wt)
    if batch_size > chunk_size:
        all_preds = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            wt_chunk = wt[i:end_idx]
            mut_chunk = mut[i:end_idx]
            pred_chunk = model(wt_chunk, mut_chunk)
            all_preds.append(pred_chunk)
        
        pred = torch.cat(all_preds, dim=0)
    else:
        pred = model(wt, mut)

    # MSE loss over entire batch
    loss_mse = F.mse_loss(pred, ddg)

    if loss_type=='mse':
        loss = loss_mse
    else:
        # Soft Spearman loss per protein
        rank_losses = []
        unique_prots = set(prot_ids)
    
        for p in unique_prots:
            # Get indices for this protein
            mask = torch.tensor([pid == p for pid in prot_ids], device=pred.device)
            n_samples = mask.sum().item()
            
            if n_samples < 2:
                continue
    
            # Compute Spearman loss for this protein
            pred_p = pred[mask]
            ddg_p = ddg[mask]
            rank_losses.append(soft_spearman_loss(pred_p, ddg_p, tau))
    
        # Average Spearman loss across proteins
        if len(rank_losses) > 0:
            loss_rank = torch.stack(rank_losses).mean()
        else:
            loss_rank = torch.tensor(0.0, device=pred.device)
    
        # Combined loss
        loss = loss_mse + lambda_rank * loss_rank
        
        # Check for NaN before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected. MSE: {loss_mse.item()}, Rank: {loss_rank.item()}")
            # Skip this batch
            return {
                "total": 0.0,
                "mse": loss_mse.item() if not torch.isnan(loss_mse) else 0.0,
                "spearman": loss_rank.item() if not torch.isnan(loss_rank) else 0.0,
                "n_proteins_with_rank": len(rank_losses)
            }
    
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), max_norm=1.0)
    optimizer.step()
    if loss_type == 'mse':
        return {
            "total": loss.item(),
            "mse": loss_mse.item()
        }
    else:
        return {
            "total": loss.item(),
            "mse": loss_mse.item(),
            "spearman": loss_rank.item() if len(rank_losses) > 0 else 0.0,
            "n_proteins_with_rank": len(rank_losses)
        }

@torch.no_grad()
def eval_step(model, dataloader):
    """Evaluation: compute average MSE"""
    model.eval()
    losses = []

    for batch in dataloader:
        wt, mut, ddg, prot_ids = batch
        ddg = ddg.cuda()
        pred = model(wt, mut)
        losses.append(F.mse_loss(pred, ddg).item())

    return sum(losses) / len(losses) if losses else 0.0

def collate_fn(batch):
    """Collate function for DataLoader"""
    return (
        [b["wt_seq"] for b in batch],
        [b["mut_seq"] for b in batch],
        torch.stack([b["ddg"] for b in batch]),
        [b["protein_id"] for b in batch],
    )

class ProteinBatchSampler(Sampler):
    """
    Custom sampler that returns all mutations from one protein per batch.
    This is crucial for computing per-protein Spearman correlation.
    """
    def __init__(self, dataset, shuffle=True):
        self.shuffle = shuffle
        self.protein_to_indices = defaultdict(list)

        for idx, row in enumerate(dataset.df.itertuples()):
            self.protein_to_indices[row.protein_id].append(idx)

        self.proteins = list(self.protein_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.proteins)

        for p in self.proteins:
            yield self.protein_to_indices[p]

    def __len__(self):
        return len(self.proteins)


df = pd.read_csv("./evaluate/ddt/S8754.csv")
df["protein_id"] = df["name"].apply(extract_protein_id)
# Split by protein (to avoid data leakage)
proteins = df["protein_id"].unique()
train_prots, val_prots = train_test_split(
    proteins, test_size=0.2, random_state=42
)
train_df = df[df["protein_id"].isin(train_prots)]
val_df = df[df["protein_id"].isin(val_prots)]
logger.info(f"Training proteins: {len(train_prots)}")
logger.info(f"Validation proteins: {len(val_prots)}")
logger.info(f"Training samples: {len(train_df)}")
logger.info(f"Validation samples: {len(val_df)}")
# Create datasets
train_dataset = StabilityDataset(train_df)
val_dataset = StabilityDataset(val_df)
# Create data loaders with protein-wise batching
if args.loss == 'mse':
    train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=32,
                          shuffle=False, collate_fn=collate_fn)
else:
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ProteinBatchSampler(train_dataset, shuffle=True),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ProteinBatchSampler(val_dataset, shuffle=False),
        collate_fn=collate_fn
    )

encoder = create_encoder_function(args.model, batch_converter)  # Replace with your encoder
embed_dim = 1280  # Replace with your embedding dimension
# Create model
model = StabilityModel(encoder, embed_dim, freeze_encoder=True).cuda()

# Optimizer (only train MLP head if encoder is frozen)
optimizer = torch.optim.Adam(model.mlp.parameters(), lr=1e-3)
# Training loop
num_epochs = 20
lambda_rank = 0.5  # Weight for Spearman loss
tau = 0.5  # Temperature for soft ranking

for epoch in range(1, num_epochs + 1):
    train_losses = []
    train_mse = []
    train_spearman = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        # Move ddG to GPU
        batch = (batch[0], batch[1], batch[2].cuda(), batch[3])
        loss_dict = train_step(model, batch, optimizer, 
                              lambda_rank=lambda_rank, tau=tau)
        
        train_losses.append(loss_dict["total"])
        train_mse.append(loss_dict["mse"])
        if args.loss=='mse':
            pbar.set_postfix({
                'loss': f'{loss_dict["total"]:.4f}',
                'mse': f'{loss_dict["mse"]:.4f}'
            })
    
            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch} Batch {batch_idx+1}: "
                          f"Loss={loss_dict['total']:.4f}, "
                          f"MSE={loss_dict['mse']:.4f}, ")
        else:
            train_spearman.append(loss_dict["spearman"])
            # Update progress bar with current losses
            pbar.set_postfix({
                'loss': f'{loss_dict["total"]:.4f}',
                'mse': f'{loss_dict["mse"]:.4f}',
                'sp': f'{loss_dict["spearman"]:.4f}',
                'n_prot': loss_dict["n_proteins_with_rank"]
            })
    
            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch} Batch {batch_idx+1}: "
                          f"Loss={loss_dict['total']:.4f}, "
                          f"MSE={loss_dict['mse']:.4f}, "
                          f"Spearman={loss_dict['spearman']:.4f}, "
                          f"N_proteins={loss_dict['n_proteins_with_rank']}")
                
    val_mse = eval_step(model, val_loader)
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_mse = sum(train_mse) / len(train_mse)
    if args.loss=='mse':
        logger.info("="*60)
        logger.info(f"Epoch {epoch:03d} Summary:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Train MSE: {avg_train_mse:.4f}")
        logger.info(f"  Val MSE: {val_mse:.4f}")
        logger.info("="*60)
    else:
        avg_train_spearman = sum(train_spearman) / len(train_spearman)
    
        logger.info("="*60)
        logger.info(f"Epoch {epoch:03d} Summary:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Train MSE: {avg_train_mse:.4f}")
        logger.info(f"  Train Spearman: {avg_train_spearman:.4f}")
        logger.info(f"  Val MSE: {val_mse:.4f}")
        logger.info("="*60)

logger.info("\nTraining completed!")

model_path = 'ddt_model_'+log_filename.split(".")[0]+'.pt'
torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
###################### evaluate
import torch
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

@torch.no_grad()
def collect_predictions(model, dataloader):
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_targets = []
    all_proteins = []

    for batch in dataloader:
        wt, mut, ddt, prot_ids = batch
        ddt = ddt.to(device)

        preds = model(wt, mut)

        all_preds.append(preds.cpu())
        all_targets.append(ddt.cpu())
        all_proteins.extend(prot_ids)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    return all_preds, all_targets, all_proteins

def protein_wise_spearman(preds, targets, proteins):
    """
    Returns:
        dict: protein_id -> Spearman rho
        float: average Spearman across proteins
    """
    by_protein = defaultdict(list)

    for p, y_hat, y in zip(proteins, preds, targets):
        by_protein[p].append((y_hat, y))

    protein_rhos = {}

    for p, values in by_protein.items():
        if len(values) < 2:
            continue

        pred_p, true_p = zip(*values)
        rho, _ = spearmanr(pred_p, true_p)

        # Handle NaN (can happen if predictions are constant)
        if not np.isnan(rho):
            protein_rhos[p] = rho

    avg_rho = np.mean(list(protein_rhos.values())) if protein_rhos else np.nan

    return protein_rhos, avg_rho

def global_spearman(preds, targets):
    rho, _ = spearmanr(preds, targets)
    return rho

def evaluate_model(model, test_loader):
    preds, targets, proteins = collect_predictions(model, test_loader)

    # Protein-wise
    protein_rhos, avg_protein_rho = protein_wise_spearman(
        preds, targets, proteins
    )

    # Global
    global_rho = global_spearman(preds, targets)

    return {
        "protein_wise": protein_rhos,
        "avg_protein_spearman": avg_protein_rho,
        "global_spearman": global_rho
    }

df_test = pd.read_csv("./evaluate/ddt/S669.csv")
df_test["protein_id"] = df_test["name"].apply(extract_protein_id)
test_dataset = StabilityDataset(df_test)
test_loader = DataLoader(
    test_dataset, 
    batch_size=32,
    collate_fn=collate_fn
)

results = evaluate_model(model, test_loader)

logger.info(f"Average protein-wise Spearman: {results['avg_protein_spearman']:.3f}")
logger.info(f"Global Spearman: {results['global_spearman']:.3f}")