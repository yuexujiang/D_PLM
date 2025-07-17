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


def vectors_to_contact_maps(vecs):
    """
    Converts multiple vectors of pairwise contact values into symmetric contact map matrices.
    
    Args:
        vecs (np.ndarray): 2D array of shape (n, L),
                           where L = N*(N-1)/2 and n is number of vectors.
    
    Returns:
        contact_maps (np.ndarray): 3D array of shape (n, N, N).
    """
    n, num_pairs = vecs.shape
    # 计算N
    N = int((1 + math.sqrt(1 + 8 * num_pairs)) / 2)
    
    # 初始化输出数组
    contact_maps = np.zeros((n, N, N), dtype=vecs.dtype)
    
    # 获取上三角索引（不包括对角线）
    triu_indices = np.triu_indices(N, k=1)
    
    # 给所有矩阵的上三角赋值
    contact_maps[:, triu_indices[0], triu_indices[1]] = vecs
    
    # 镜像到下三角，使矩阵对称
    contact_maps = contact_maps + np.transpose(contact_maps, (0, 2, 1))
    
    return contact_maps
    

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

def normalize_contact_map(contact_map):
    # Option 1: Divide by maximum value (if maximum is known and consistent)
    max_val = contact_map.max()  # or a fixed value like 6.28 if appropriate
    normalized_map = contact_map / max_val
    # Optionally clip to [0,1] to ensure all values are within range
    normalized_map = np.clip(normalized_map, 0, 1)
    return normalized_map


def process_samples(data_folder2search, model_name, sub_list, output_folder):
    image_processor = VivitImageProcessor.from_pretrained(model_name)
    model = VivitModel.from_pretrained(model_name)
    # Specify the parent folder (folder A)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VivitModel.from_pretrained(model_name).to(device)
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
            if len(sequence)>512:
                continue
            #fragments, frag_contact = fragment_sequence(sequence, np.array([]), fixed_length=224, overlap=30)
            #data_len=len(fragments)
            
            for file_xtc_path in traj_files:
                file_path_base = os.path.splitext(os.path.basename(file_xtc_path))[0]
                #data[pid]={'ind2seq':{}, 'ind2map':{}}
                #for i in range(data_len):
                #     data[pid]['ind2map'][i]=[]

                traj = md.load(file_xtc_path, top=file_pdb_path)
                distances = md.compute_distances(traj, traj.topology.select_pairs('name CA', 'name CA'))
                contact_maps = vectors_to_contact_maps(distances) # [n, N, N]
                contact_maps = np.expand_dims(contact_maps, axis=-1) # [n, N, N, 1]
                contact_maps = normalize_contact_map(contact_maps) # [n, N, N, 1]
                contact_maps = np.repeat(contact_maps, 3, axis=-1) # [n, N, N, 3]

                images = image_processor(list(contact_maps), return_tensors="pt", do_rescale=False, offset=False, 
                                         do_normalize=False, do_resize=True, size=224, do_center_crop=False) # [1, 1000, 3, 224, 224]
                images = images['pixel_values']
                #print(images.dtype)
                groups = group_frames(images)
                #print(groups.dtype)
                groups = groups.to(device)
                sub_num=groups.shape[0]//8
                sub_remain=groups.shape[0]%8
                rep_list=[]
                print(get_memory_usage_gb())
                for z in range(sub_num):
                    #print(z)
                    sub_groups = groups[z*8:z*8+8]
                    with torch.no_grad():
                        vivit_output = model(sub_groups)
                    last_hidden_states = vivit_output.last_hidden_state #[list_len, 3137, 768]
                    cls_representation = last_hidden_states[:, 0, :] #[list_len, 1, 768]
                    # sub_list.append(cls_representation)
                    rep_list.append(cls_representation.detach().cpu())
                    #print(get_memory_usage_gb())
                
                #cls_tem=torch.cat(rep_list, dim=0) #[list_len, 1, 768]
                if sub_remain!=0:
                    sub_groups = groups[-sub_remain:]
                    with torch.no_grad():
                        vivit_output = model(sub_groups)
                    last_hidden_states = vivit_output.last_hidden_state #[list_len, 3137, 768]
                    cls_representation = last_hidden_states[:, 0, :] #[list_len, 1, 768]
                    rep_list.append(cls_representation.detach().cpu())
                        
                    
                #cls_representation = torch.stack(rep_list, axis=0) #[list_len,  768]
                cls_representation = torch.cat(rep_list, dim=0) #[list_len, 768]
                #print(cls_representation.shape)
                # pooled = cls_representation.mean(axis=0)  # shape: (768,)
                pooled = torch.cat([cls_representation.mean(0),
                                    cls_representation.std(0),
                                    cls_representation.max(0).values], dim=0)  # → shape: [768 * 3]
                pooled = pooled.reshape(-1)

                #print(pooled.shape)
                # samples.append((h5id, pid_i, seq_i, pooled))
                outputfile = os.path.join(output_folder, f"{file_path_base}.h5")
                if os.path.exists(outputfile):
                    continue
                print(f'processing {outputfile}')
                #data[pid]={'ind2seq':{}, 'ind2map':{}}
                #for i in range(data_len):
                #     data[pid]['ind2map'][i]=[]
                os.makedirs(os.path.dirname(outputfile), exist_ok=True)
                write_h5_file(outputfile, pooled, pid, sequence)


                del rep_list, images, groups
                torch.cuda.empty_cache()
                print(get_memory_usage_gb())
                del traj
                torch.cuda.empty_cache()
                print(get_memory_usage_gb())
    


def write_h5_file(file_path, geom2vec, pid, seq):
    with h5py.File(file_path, 'w') as f:
        # f.create_dataset('seq', data=seq)
        f.create_dataset('geom2vec', data=geom2vec)
        f.create_dataset('pid', data=pid)
        f.create_dataset('seq', data=str(seq).encode('utf-8'))

def loadh5(h5file):
    with h5py.File(h5file, 'r') as f:
        name = os.path.basename(h5file)
        pid = "#".join(name.split('#')[:2])
        # pid=f[pid].attrs['pid']
        # pid = 'some_pid'
        rep = f[pid]['pooled'][:]
        seq = f[pid].attrs['seq']
    return pid, seq, rep


def group_frames(traj, group_size=32): # traj => [1, 1001, 3, 224, 224]
    """
    Group frames into sets of specified size.
    
    Args:
        total_frames (int): Total number of frames
        group_size (int): Number of frames in each group
        
    Returns:
        list: List of lists, where each inner list is a group of frames
    """
    total_frames = traj.shape[1]
    groups = []
    
    # Calculate how many complete groups we'll have
    num_complete_groups = total_frames // group_size
    
    # Group the frames
    for i in range(num_complete_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group = traj[:,start_idx:end_idx,:,:,:]
        groups.append(group)
    
    # Handle any remaining frames
    remaining_frames = total_frames % group_size
    if remaining_frames > 0:
        group = traj[:,total_frames-32:,:,:,:]
        groups.append(group)
    
    processed_arrays = [arr.squeeze(0) for arr in groups]  # shape becomes [32, 3, 224, 224]
    # Stack along a new axis (axis=0)
    result = torch.stack(processed_arrays, axis=0) # [list_len, 32, 3, 224, 224]
    return result

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--data_folder2search", help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/')
    parser.add_argument("--model_name", default='google/vivit-b-16x2-kinetics400', help='xx')
    parser.add_argument("--list_num", type=int, default=0, help="xx")
    parser.add_argument("--output_folder", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_data', help="xx")
    parser.add_argument("--split_num", type=int, default=6, help="xx")

    
    args_main = parser.parse_args()
    lists = split_subfolders_into_n_lists(args_main.data_folder2search, n=args_main.split_num)
    process_samples(args_main.data_folder2search, args_main.model_name,
                     sub_list=lists[args_main.list_num], output_folder=args_main.output_folder)
