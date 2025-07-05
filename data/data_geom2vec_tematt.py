import torch
import os
import mdtraj as md
import numpy as np
import math
from Bio.PDB import PDBParser, PPBuilder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from transformers import VivitImageProcessor, VivitModel
import h5py
from .utils import get_gpu_usage_smi, get_memory_usage_gb
import argparse


class ProteinMDDataset(Dataset):
    def __init__(self, samples, configs,mode="train"):
        self.original_samples = samples
    def __len__(self):
           return len(self.original_samples)
    def __getitem__(self, idx):
        pid, seq, res_rep = self.original_samples[idx]
        return pid, seq, res_rep #res_rep [n_residue, T, 512]

import torch.nn.functional as F

def bin_average_functional(tensor, n_bins=300):
    """
    tensor: shape [sum_n_residue, 10001, 512]
    Returns: shape [sum_n_residue, n_bins, 512]
    """
    tensor = tensor.cpu()  # Move to CPU to save GPU memory

    batch_size, seq_len, hidden_dim = tensor.shape
    
    # Use adaptive average pooling
    # Reshape to [batch_size * hidden_dim, 1, seq_len] for pooling
    reshaped = tensor.permute(0, 2, 1).contiguous()  # [batch_size, hidden_dim, seq_len]
    reshaped = reshaped.view(-1, 1, seq_len)  # [batch_size * hidden_dim, 1, seq_len]
    
    # Apply adaptive average pooling
    pooled = F.adaptive_avg_pool1d(reshaped, n_bins)  # [batch_size * hidden_dim, 1, n_bins]
    
    # Reshape back
    pooled = pooled.view(batch_size, hidden_dim, n_bins)
    output = pooled.permute(0, 2, 1)  # [batch_size, n_bins, hidden_dim]
    
    return output

def custom_collate(batch):
    pid, seq, res_rep = zip(*batch)
    # prot_vec = torch.stack(prot_vec, dim=0)  # shape: [batch_size, 1024]

    res_reps = list(res_rep) # list of [n_residue_i, T, 512] tensors
    # Flatten all residues into one tensor
    flat_res_rep = torch.cat(res_reps, dim=0)  # shape: [sum_n_residue, T, 512]
    
    # Build a batch index to keep track of which residue belongs to which protein
    batch_idx = torch.cat([
        torch.full((r.shape[0],), i, dtype=torch.long)
        for i, r in enumerate(res_reps)
    ])  # shape: [sum_n_residue]
    
    batched_data = {'pid': pid, 'seq': seq, 'res_rep': flat_res_rep, 'batch_idx': batch_idx} #bin_res_rep [sum_n_residue, T, 512]
    # return pid, seq, contacts
    return batched_data

def prepare_replicate(configs, train_repli_path, test_repli_path):
    samples = prepare_samples(train_repli_path, configs)
    total_samples = len(samples)
    val_size = int(total_samples * 0.1)
    test_size = int(total_samples * 0.1)
    train_size = total_samples - val_size - test_size
    train_samples, val_samples, test_samples = random_split(samples, [train_size, val_size, test_size])

    samples_hard = prepare_samples(test_repli_path, configs)
    hard_num = len(samples_hard)
    val_size = int(hard_num * 0.5)
    test_size = hard_num - val_size
    val_hard, test_hard = random_split(samples_hard, [val_size, test_size])

    val_samples = val_samples + val_hard
    test_samples = test_samples + test_hard
    print(f"train samples: {len(train_samples)}, val samples: {len(val_samples)}, test samples: {len(test_samples)}")
    # Create DataLoader for each split
    train_dataset = ProteinMDDataset(train_samples, configs=configs, mode="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=False
    )
    val_dataset = ProteinMDDataset(val_samples, configs=configs, mode="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=custom_collate,
        drop_last=False
    )
    test_dataset = ProteinMDDataset(test_samples, configs=configs, mode="val")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=custom_collate,
        drop_last=False
    )
    # dataset = ProteinMDDataset(samples, configs=configs, mode = "train")
    # dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate,drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader

def prepare_dataloaders(configs):
    train_dataloader, val_dataloader, test_dataloader = prepare_replicate(configs, 
                                                                          configs.train_settings.Atlas_data_path, 
                                                                          configs.train_settings.Atlas_test_path)
  
    return train_dataloader, val_dataloader, test_dataloader

def chunk_bin(input_tensor, chunk_size=100, bin_number=100):
    outputs = []
    for chunk in torch.split(input_tensor, chunk_size, dim=0):  # Process 1000 residues at a time
        chunk = bin_average_functional(chunk, n_bins=bin_number)
        outputs.append(chunk)
    return torch.cat(outputs, dim=0)

def prepare_samples(datapath, configs):
    maxlen = configs.model.esm_encoder.max_length
    samples=[]
    for file_name in os.listdir(datapath):
        if file_name.endswith(".pt"):
            file_path = os.path.abspath(os.path.join(datapath, file_name))
            traj_rep = torch.load(file_path) # [T, M, 4, D]
            traj_rep = traj_rep.permute(1, 0, 2, 3) #[M, T, 4, D]
            traj_rep = traj_rep.reshape(traj_rep.shape[0], traj_rep.shape[1], -1)  # → [M, T, 4*D]
            # bin_res_rep = bin_average_functional(traj_rep, n_bins=100) #[M, 100, dim]
            bin_res_rep = chunk_bin(traj_rep, chunk_size=100, bin_number=100) #[M, 100, dim]

            basename = os.path.basename(file_name)
            without_extension=os.path.splitext(basename)[0]
            h5_filename = os.path.join(datapath, f"{without_extension}.h5")
            geom2vec, seq, pid = load_h5(h5_filename)
        else:
            continue
        # prot_rep = geom2vec.mean
        samples.append((pid, seq, bin_res_rep[:maxlen])) #bin_res_rep [n_residue, 1000, dim]
    
    return samples


def load_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        geom2vec = f['geom2vec'][:]
        pid = f['pid'][()].decode('utf-8')
        seq = f['seq'][()].decode('utf-8')  # Decode from bytes to string
    
    return geom2vec, seq, pid

  



