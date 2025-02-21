import torch
import os
import mdtraj as md
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from torch.utils.data import Dataset
import webdataset as wds
from torch.utils.data import DataLoader

class ProteinMDDataset(Dataset):
    def __init__(self, samples, configs,mode="train"):
        self.original_samples = samples
    def __len__(self):
           return len(self.original_samples)
    def __getitem__(self, idx):
        seq, contact = self.samples[idx]
        return seq, contact


def custom_collate(batch):
    seq, contact= zip(*batch)
    return seq, contact

def prepare_dataloaders(configs):
    samples = prepare_samples(configs.train_settings.MD_data_path)
    dataset = ProteinMDDataset(samples, configs=configs, mode = "train")
    dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate,drop_last=False)
    return dataloader



def vector_to_contact_map(vec):
    """
    Converts a vector of pairwise contact values into a symmetric contact map matrix.
    
    Args:
        vec (np.ndarray): 1D array of length L, where L = N*(N-1)/2.
    
    Returns:
        contact_map (np.ndarray): 2D symmetric array of shape (N, N).
    """
    num_pairs = vec.shape[0]
    # Calculate N using the quadratic formula: N = (1 + sqrt(1+8L)) / 2
    N = int((1 + math.sqrt(1 + 8 * num_pairs)) / 2)
    
    # Initialize an empty contact map matrix
    contact_map = np.zeros((N, N))
    
    # Get indices for the upper triangle (excluding the diagonal)
    triu_indices = np.triu_indices(N, k=1)
    
    # Fill the upper triangle with the vector values
    contact_map[triu_indices] = vec
    
    # Mirror the upper triangle to the lower triangle to make it symmetric
    contact_map = contact_map + contact_map.T
    
    return contact_map

def pdb2seq(pdb_path):
    # Create a PDB parser object
    parser = PDBParser(QUIET=True)
    
    # Parse the structure from the PDB file
    structure = parser.get_structure("protein", pdb_path)
    
    # Initialize the polypeptide builder
    ppb = PPBuilder()
    
    # Extract sequences from all chains in the structure
    for model in structure:
        for chain in model:
            # Build polypeptides for the chain (could be more than one segment)
            polypeptides = ppb.build_peptides(chain)
            for poly_index, poly in enumerate(polypeptides):
                sequence = poly.get_sequence()
                # print(f"Chain {chain.id} (segment {poly_index}): {sequence}")
    return sequence

def prepare_samples(data_folder2search):
    # Specify the parent folder (folder A)
    folder_A = data_folder2search
    samples = []
    
    # Iterate through each subfolder under folder_A
    for subfolder in os.listdir(folder_A):
        subfolder_path = os.path.join(folder_A, subfolder)
        
        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            num_xtc=0
            num_pdb=0

            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".xtc"):
                    file_xtc_path = os.path.join(subfolder_path, file_name)
                    num_xtc += 1
                elif file_name.endswith(".pdb"):
                    file_pdb_path = os.path.join(subfolder_path, file_name)
                    num_pdb += 1
            
            if num_xtc==num_pdb==1:
                sequence = pdb2seq(file_pdb_path)
                traj = md.load(file_xtc_path, top=file_pdb_path)
                # cutoff = 7  # in nanometers
                contact_maps = []
                for frame in traj:
                    # compute pairwise distances
                    distances = md.compute_distances(frame, traj.topology.select_pairs('name CA', 'name CA')) # [1, L] 
                    # generate contact map: 1 if distance < cutoff, else 0
                    # contact_map = (distances < cutoff).astype(int)
                    contact_map = vector_to_contact_map(distances.reshape(-1)) # [N, N]
                    contact_maps.append(contact_map)

                samples.append((sequence, contact_maps))
    return samples




