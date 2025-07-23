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
        pid, seq, rep = self.original_samples[idx]
        return pid, seq, rep


def custom_collate(batch):
    pid, seq, rep = zip(*batch)
    # Remove the extra dimension from each contact sample
    # traj_list = [traj.squeeze(0) for traj in reps]
    # Now each traj has shape [32, 3, 224, 224]
    # Stack them along a new first dimension (batch dimension)
    # traj_batch = torch.stack(traj_list, dim=0)
    batched_data = {'pid': pid, 'seq': seq, 'traj': rep}
    # return pid, seq, contacts
    return batched_data

def prepare_replicate(configs, train_repli_path, test_repli_path):
    if configs.model.MD_encoder.meanrep:
        samples = prepare_samples_meanrep(train_repli_path)
    else:
        samples = prepare_samples(train_repli_path)
    total_samples = len(samples)
    val_size = int(total_samples * 0.1)
    test_size = int(total_samples * 0.1)
    train_size = total_samples - val_size - test_size
    train_samples, val_samples, test_samples = random_split(samples, [train_size, val_size, test_size])

    # samples_hard = prepare_samples(test_repli_path)
    if configs.model.MD_encoder.meanrep:
        samples_hard = prepare_samples_meanrep(test_repli_path)
    else:
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


def prepare_samples(datapath):
    samples=[]
    for file_name in os.listdir(datapath):
        file_path = os.path.abspath(os.path.join(datapath, file_name))
        pid, seq, rep = load_h5(file_path)
        samples.append((pid, seq, rep))
    
    return samples

def prepare_samples_meanrep(datapath):
    samples=[]
    for file_name in os.listdir(datapath):
        file_path = os.path.abspath(os.path.join(datapath, file_name))
        pid, seq, rep = load_h5(file_path)
        rep = rep[:768]
        samples.append((pid, seq, rep))
    
    return samples

def load_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        pooled = f['geom2vec'][:]
        pid = f['pid'][()].decode('utf-8')
        seq = f['seq'][()].decode('utf-8')  # Decode from bytes to string
    
    return pid, seq, pooled




