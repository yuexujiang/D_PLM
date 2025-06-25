from torch_geometric.data import DataLoader, Batch
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import h5py
from torch_geometric.data import Data

def build_dccm_graph(dccm_matrix: np.ndarray, 
                     mi_matrix: np.ndarray, 
                     rmsf: torch.Tensor,
                     top_k: int = 15) -> Data:
    """
    Build PyTorch Geometric graph from DCCM and MI matrices
    
    Args:
        dccm_matrix: DCCM matrix of shape [n_residue, n_residue]
        mi_matrix: MI matrix of shape [n_residue, n_residue]
        rmsf: RMSF tensor of shape [n_residue, 10000]
        top_k: Number of top neighbors to select for each residue
    
    Returns:
        PyTorch Geometric Data object
    """
    n_residue = dccm_matrix.shape[0]
    
    # Find top-k neighbors for each residue
    edge_list = []
    edge_features = []
    
    for i in range(n_residue):
        # Get DCCM values for residue i (excluding self-connection)
        dccm_row = dccm_matrix[i].copy()
        dccm_row[i] = -np.inf  # Exclude self-connection
        
        # Find top-k neighbors
        top_indices = np.argsort(dccm_row)[-top_k:]
        
        for j in top_indices:
            # Add edge from i to j
            edge_list.append([i, j])
            
            # Edge features: [DCCM_value, MI_value]
            edge_feat = [dccm_matrix[i, j], mi_matrix[i, j]]
            edge_features.append(edge_feat)

            
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    node_features = torch.tensor(rmsf, dtype=torch.float32)
    
    # Create PyTorch Geometric data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

# Batch processing utilities


class ProteinDataset(Dataset):
    """
    Dataset class for handling multiple protein samples with different residue numbers
    """
    def __init__(self, dccm_matrices, mi_matrices, rmsf_tensors, seqs, pids, labels=None, top_k=15):
        """
        Args:
            dccm_matrices: List of DCCM matrices, each with shape [n_residue_i, n_residue_i]
            mi_matrices: List of MI matrices, each with shape [n_residue_i, n_residue_i]
            rmsf_tensors: List of RMSF tensors, each with shape [n_residue_i, 10000]
            labels: Optional list of labels for supervised learning
            top_k: Number of neighbors for each residue
        """
        assert len(dccm_matrices) == len(mi_matrices) == len(rmsf_tensors)
        
        self.dccm_matrices = dccm_matrices
        self.mi_matrices = mi_matrices
        self.rmsf_tensors = rmsf_tensors
        self.seqs = seqs
        self.pids = pids
        self.labels = labels
        self.top_k = top_k
        
    def __len__(self):
        return len(self.dccm_matrices)
    
    def __getitem__(self, idx):
        # Build graph for sample idx
        graph_data = build_dccm_graph(
            self.dccm_matrices[idx],
            self.mi_matrices[idx], 
            self.rmsf_tensors[idx],
            top_k=self.top_k
        )
        seq = self.seqs[idx]
        pid = self.pids[idx]
        
        # Add label if available
        if self.labels is not None:
            graph_data.y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return graph_data, seq, pid

def custom_collate(batch):
    """
    Custom collate function using torch_geometric.data.Batch
    
    Args:
        batch: List of Data objects from dataset
    
    Returns:
        Batched Data object
    """
    torch_geometric_feature = [item[0] for item in batch]  # item[0] is for torch_geometric Data

    # Create a Batch object
    torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    raw_seqs = [item[1] for item in batch]
    pids = [item[2] for item in batch]
    batched_data = {'graph': torch_geometric_batch, 'seq': raw_seqs, 'pid': pids}
    return batched_data
    # return Batch.from_data_list(batch)


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
    temp = list(zip(*train_samples))
    rmsf_list, dccm_list, mi_list, seq_list, pid_list = [list(x) for x in temp]
    train_dataset = ProteinDataset(dccm_list, mi_list, rmsf_list, seq_list, pid_list)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=False
    )
    temp = list(zip(*val_samples))
    rmsf_list, dccm_list, mi_list, seq_list, pid_list = [list(x) for x in temp]
    val_dataset = ProteinDataset(dccm_list, mi_list, rmsf_list, seq_list, pid_list)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=custom_collate,
        drop_last=False
    )
    temp = list(zip(*test_samples))
    rmsf_list, dccm_list, mi_list, seq_list, pid_list = [list(x) for x in temp]
    test_dataset = ProteinDataset(dccm_list, mi_list, rmsf_list, seq_list, pid_list)
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
        rmsf, dccm, mi, seq, pid = load_h5_file_v2(file_path)

        rmsf = torch.tensor(rmsf)
        chunk_size = 100
        means = []
        for i in range(0, rmsf.shape[1], chunk_size):
            chunk = rmsf[:, i:i+chunk_size]  # shape: [512, <=100]
            chunk_mean = chunk.mean(dim=1, keepdim=True)  # mean over columns
            means.append(chunk_mean)
        
        # Concatenate means along the second dimension
        rmsf = torch.cat(means, dim=1)  # shape: [512, num_chunks]
        samples.append((rmsf[:maxlen], dccm[:maxlen, :maxlen], mi[:maxlen, :maxlen], seq, pid))
    
    return samples

def load_h5_file_v2(file_path):
    with h5py.File(file_path, 'r') as f:
        rmsf = f['rmsf'][:]
        dccm = f['dccm'][:]
        mi = f['mi'][:]
        pid = f['pid'][()].decode('utf-8')
        seq = f['seq'][()].decode('utf-8')  # Decode from bytes to string
        # seq = f['seq']
        # pid = f['pid']
    
    return rmsf, dccm, mi, seq, pid