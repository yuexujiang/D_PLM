import MDAnalysis as mda
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import pdist, squareform
import mdtraj as md
from Bio.PDB import PDBParser, PPBuilder
import os
import h5py
import argparse


def load_trajectory(topology_file, xtc_file):
    u = mda.Universe(topology_file, xtc_file)
    ca_atoms = u.select_atoms("protein and name CA")
    n_residues = len(ca_atoms)
    n_frames = len(u.trajectory)
    positions = np.zeros((n_frames, n_residues, 3))
    for i, ts in enumerate(u.trajectory):
        positions[i] = ca_atoms.positions
    return positions

def compute_dccm(positions):
    # Subtract mean position to get displacements
    disp = positions - positions.mean(axis=0, keepdims=True)
    n_frames, n_residues, _ = disp.shape
    dccm = np.zeros((n_residues, n_residues))
    for i in range(n_residues):
        vi = disp[:, i, :]
        vi_dot = np.sum(vi * vi, axis=1)
        for j in range(i, n_residues):
            vj = disp[:, j, :]
            dot_prod = np.sum(vi * vj, axis=1)
            numerator = np.sum(dot_prod)
            denom = np.sqrt(np.sum(vi_dot) * np.sum(np.sum(vj * vj, axis=1)))
            dccm[i, j] = dccm[j, i] = numerator / denom if denom > 0 else 0.0
    
    return dccm

def compute_mi(positions, bins=20):
    # Use L2 norm of displacement vectors
    disp = positions - positions.mean(axis=0, keepdims=True)
    magnitudes = np.linalg.norm(disp, axis=2)  # shape: [frames, residues]
    # Bin the magnitudes
    mags_min = np.min(magnitudes)
    mags_max = np.max(magnitudes)
    norm_mags = ((magnitudes - mags_min) / (mags_max - mags_min + 1e-8) * bins).astype(int)
    n_frames, n_residues = norm_mags.shape
    mi_matrix = np.zeros((n_residues, n_residues))
    for i in range(n_residues):
        for j in range(i, n_residues):
            mi = mutual_info_score(norm_mags[:, i], norm_mags[:, j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    
    return mi_matrix

def compute_se3_dccm_and_mi(topology_file, xtc_file):
    positions = load_trajectory(topology_file, xtc_file)
    dccm = compute_dccm(positions)
    mi = compute_mi(positions)
    return dccm, mi

def write_h5_file(file_path, rmsf, dccm, mi, pid, seq):
    with h5py.File(file_path, 'w') as f:
        # f.create_dataset('seq', data=seq)
        f.create_dataset('rmsf', data=rmsf)
        f.create_dataset('dccm', data=dccm)
        f.create_dataset('mi', data=mi)
        f.create_dataset('pid', data=pid)
        f.create_dataset('seq', data=str(seq).encode('utf-8'))



def calculate_rmsf(file_pdb_path, file_xtc_path):
    traj = md.load(file_xtc_path, top=file_pdb_path)
    # Calculate the number of residues and atoms
    n_residues = traj.topology.n_residues
    num_frames = traj.n_frames
    # Initialize a list to store RMSF for each residue
    rmsf = np.zeros((n_residues, num_frames))
    # Calculate the average positions of each residue (CA atom positions)
    avg_positions = np.mean(traj.xyz, axis=0)  # Shape: (n_atoms, 3)
    # Loop through each residue and calculate RMSF
    for i, residue in enumerate(traj.topology.residues):
        # Select all atoms of the residue (typically just the CA atom)
        atoms = residue.atoms
        atom_indices = [atom.index for atom in atoms]
        # Extract the positions for these atoms over all frames
        atom_positions = traj.xyz[:, i, :]  # Shape: (n_frames, n_atoms, 3)
        # Calculate the deviation from the average position
        deviations = atom_positions - avg_positions[i, :]
        # Compute RMSF for the residue (mean squared deviations)
        # rmsf[i] = np.sqrt(np.mean(np.sum(deviations ** 2, axis=1)))
        rmsf[i] = np.sqrt(np.sum(deviations ** 2, axis=1))
    
    #rmsf [n_residue, frame]
    return rmsf

# def dccm_to_edge_index(dccm, threshold):
#     """
#     Converts a DCCM matrix to an edge_index tensor for values above a threshold.

#     Args:
#         dccm (np.ndarray): NxN symmetric matrix with dynamic cross-correlation coefficients.
#         threshold (float): Correlation cutoff. Only values above this will be included as edges.

#     Returns:
#         edge_index (np.ndarray): Array of shape [2, edge_num] with source and target indices.
#     """
#     assert dccm.shape[0] == dccm.shape[1], "DCCM must be square"
#     n = dccm.shape[0]
#     # Get indices where DCCM ≥ threshold (and exclude self-loops)
#     src, tgt = np.where((dccm >= threshold) & (np.eye(n) == 0))
#     # Stack into edge_index shape: [2, edge_num]
#     edge_index = np.stack([src, tgt], axis=0)
#     return edge_index
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

def process_samples(data_folder2search, sub_list, output_folder):
    # Specify the parent folder (folder A)
    folder_A = data_folder2search
    samples = []
    
    # Iterate through each subfolder under folder_A
    #data={}
    for subfolder in os.listdir(folder_A):
        subfolder_path = os.path.abspath(os.path.join(folder_A, subfolder))
        
        # Check if the path is a directory
        if os.path.isdir(subfolder_path) and subfolder_path in sub_list:
            traj_files=[]
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".xtc"):
                    file_xtc_path = os.path.join(subfolder_path, file_name)
                    traj_files.append(file_xtc_path)
                    # num_xtc += 1
                elif file_name.endswith(".pdb"):
                    file_pdb_path = os.path.join(subfolder_path, file_name)
                    pid = file_name.split(".")[0]
                    # num_pdb += 1
            print(pid)
            sequence = pdb2seq(file_pdb_path)
            for file_xtc_path in traj_files:
                file_path_base = os.path.basename(file_xtc_path)
                outputfile = os.path.join(output_folder, os.path.splitext(file_path_base)[0] + f".h5")
                if os.path.exists(outputfile):
                    continue
                print(f'processing {outputfile}')
                #data[pid]={'ind2seq':{}, 'ind2map':{}}
                #for i in range(data_len):
                #     data[pid]['ind2map'][i]=[]
                dccm_matrix, mi_matrix = compute_se3_dccm_and_mi(file_pdb_path, file_xtc_path)
                rmsf = calculate_rmsf(file_pdb_path, file_xtc_path)
                os.makedirs(os.path.dirname(outputfile), exist_ok=True)
                write_h5_file(outputfile, rmsf, dccm_matrix, mi_matrix, pid, sequence)

def split_subfolders_into_n_lists(folder_path, n=6):
    # 获取folder_path下所有一级子文件夹（完整路径）
    all_subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, d))]
    # 按字母排序（可选）
    all_subfolders.sort()
    # 创建n个空列表
    lists = [[] for _ in range(n)]
    # 均匀分配
    for i, subfolder in enumerate(all_subfolders):
        lists[i % n].append(os.path.abspath(subfolder))
    
    return lists

def add_id_seq_2h5(csv_path, h5_folder_path):
    import os
    import pandas as pd
    import h5py
    
    # Step 1: Read the CSV file into a pandas DataFrame
    # csv_file = 'path_to_your_csv_file.csv'
    csv_file = csv_path
    df = pd.read_csv(csv_file)
    
    # Create a dictionary to map 'name' to 'seqres'
    name_to_sequence = dict(zip(df['name'], df['seqres']))
    
    # Step 2: Define the folder containing your H5 files
    # h5_folder = 'path_to_your_h5_folder'
    h5_folder = h5_folder_path
    
    # Step 3: Iterate through each H5 file in the folder
    for h5_filename in os.listdir(h5_folder):
        if h5_filename.endswith('.h5'):
            # Extract the 'name' part of the filename (e.g., 'xx_YY' from 'xx_YY_aa_bb.h5')
            name_part = '_'.join(h5_filename.split('_')[:2])  # Assuming format xx_YY_aa_bb.h5
            
            # Step 4: Check if the name exists in the CSV mapping
            if name_part in name_to_sequence:
                sequence = name_to_sequence[name_part]
                
                # Step 5: Open the H5 file and add 'name' and 'sequence' information
                h5_path = os.path.join(h5_folder, h5_filename)
                with h5py.File(h5_path, 'a') as f:  # Open in append mode
                    # Add the 'name' and 'sequence' as new datasets in the H5 file
                    if 'name' not in f:
                        f.create_dataset('pid', data=name_part)
                    if 'sequence' not in f:
                        f.create_dataset('seq', data=sequence.encode('utf-8'))  # Store sequence as bytes
                
                print(f"Added name and sequence to {h5_filename}")
            else:
                print(f"Name {name_part} not found in the CSV for {h5_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--data_folder2search", help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/')
    parser.add_argument("--list_num", type=int, default=0, help="xx")
    parser.add_argument("--output_folder", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data', help="xx")
    parser.add_argument("--split_num", type=int, default=6, help="xx")

    
    args_main = parser.parse_args()
    lists = split_subfolders_into_n_lists(args_main.data_folder2search, n=args_main.split_num)
    process_samples(args_main.data_folder2search, 
                     sub_list=lists[args_main.list_num], output_folder=args_main.output_folder)