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
        pid, seq, res_rep, prot_vec = self.original_samples[idx]
        return pid, seq, res_rep, prot_vec #res_rep [n_residue, 512], pro_vec [1024,]


def custom_collate(batch):
    pid, seq, res_rep, prot_vec = zip(*batch)
    prot_vec = torch.stack(prot_vec, dim=0)  # shape: [batch_size, 1024]

    res_reps = list(res_rep) # list of [n_residue_i, 512] tensors
    # Flatten all residues into one tensor
    flat_res_rep = torch.cat(res_reps, dim=0)  # shape: [sum_n_residue, 512]
    
    # Build a batch index to keep track of which residue belongs to which protein
    batch_idx = torch.cat([
        torch.full((r.shape[0],), i, dtype=torch.long)
        for i, r in enumerate(res_reps)
    ])  # shape: [sum_n_residue]
    
    batched_data = {'pid': pid, 'seq': seq, 'res_rep': flat_res_rep, 'batch_idx': batch_idx, 'prot_vec': prot_vec}
    # return pid, seq, contacts
    return batched_data

def prepare_replicate(configs, train_repli_path, test_repli_path):
    samples = prepare_samples(train_repli_path)
    total_samples = len(samples)
    val_size = int(total_samples * 0.1)
    test_size = int(total_samples * 0.1)
    train_size = total_samples - val_size - test_size
    train_samples, val_samples, test_samples = random_split(samples, [train_size, val_size, test_size])

    samples_hard = prepare_samples(test_repli_path)
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
    train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0 = prepare_replicate(configs, 
                                                                          configs.train_settings.Atlas_data_repli_0_path, 
                                                                          configs.train_settings.Atlas_test_repli_0_path)
    train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1 = prepare_replicate(configs, 
                                                                          configs.train_settings.Atlas_data_repli_1_path, 
                                                                          configs.train_settings.Atlas_test_repli_1_path)
    train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2 = prepare_replicate(configs, 
                                                                          configs.train_settings.Atlas_data_repli_2_path, 
                                                                          configs.train_settings.Atlas_test_repli_2_path)
    return ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
            (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
            (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2))



def prepare_samples(datapath, configs):
    maxlen = configs.model.esm_encoder.max_length
    samples=[]
    for file_name in os.listdir(datapath):
        file_path = os.path.abspath(os.path.join(datapath, file_name))
        geom2vec, seq, pid = load_h5(file_path)
        res_rep = torch.tensor(geom2vec)
        mean = geom2vec.mean(axis=0)
        std = geom2vec.std(axis=0)
        protein_vector = np.concatenate([mean, std], axis=-1)
        protein_vector = torch.tensor(protein_vector)

        # prot_rep = geom2vec.mean
        samples.append((pid, seq, res_rep[:maxlen], protein_vector)) #res_rep [n_residue, 512], pro_vec [1024,]
    
    return samples


def load_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        geom2vec = f['geom2vec'][:]
        pid = f['pid'][()].decode('utf-8')
        seq = f['seq'][()].decode('utf-8')  # Decode from bytes to string
    
    return geom2vec, seq, pid

  



