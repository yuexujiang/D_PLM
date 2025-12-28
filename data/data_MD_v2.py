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
import torch.fft

import torch.nn as nn
from scipy.signal import find_peaks
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch_dct


class ProteinMDDataset(Dataset):
    def __init__(self, samples, configs,mode="train"):
        self.original_samples = samples
    def __len__(self):
           return len(self.original_samples)
    def __getitem__(self, idx):
        # pid, seq, rep = self.original_samples[idx]
        # return pid, seq, rep
        sample_path = self.original_samples[idx]
        pid, seq, rep = loadh5(sample_path)
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
    samples = prepare_samples(train_repli_path)
    total_samples = len(samples)
    val_size = int(total_samples * 0.1)
    test_size = int(total_samples * 0.1)
    train_size = total_samples - val_size - test_size
    train_samples, val_samples, test_samples = random_split(samples, [train_size, val_size, test_size])
    # 
    samples_hard = prepare_samples(test_repli_path)
    hard_num = len(samples_hard)
    val_size = int(hard_num * 0.5)
    test_size = hard_num - val_size
    val_hard, test_hard = random_split(samples_hard, [val_size, test_size])
    #
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
    

# def pdb2seq(pdb_path):
#     # Create a PDB parser object
#     parser = PDBParser(QUIET=True)
    
#     # Parse the structure from the PDB file
#     structure = parser.get_structure("protein", pdb_path)
    
#     # Initialize the polypeptide builder
#     ppb = PPBuilder()
    
#     # Extract sequences from all chains in the structure
#     for model in structure:
#         for chain in model:
#             # Build polypeptides for the chain (could be more than one segment)
#             polypeptides = ppb.build_peptides(chain)
#             for poly_index, poly in enumerate(polypeptides):
#                 sequence = poly.get_sequence()
#                 # print(f"Chain {chain.id} (segment {poly_index}): {sequence}")
#     return sequence
def pdb2seq(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ppb = PPBuilder()
    
    sequences = []
    for model in structure:
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            seq = "".join(str(poly.get_sequence()) for poly in polypeptides)
            sequences.append(seq)
    
    return "".join(sequences)

def normalize_contact_map(contact_map):
    # Option 1: Divide by maximum value (if maximum is known and consistent)
    max_val = contact_map.max()  # or a fixed value like 6.28 if appropriate
    normalized_map = contact_map / max_val
    # Optionally clip to [0,1] to ensure all values are within range
    normalized_map = np.clip(normalized_map, 0, 1)
    return normalized_map

def prepare_samples(datapath):
    # samples=[]
    # for file_name in os.listdir(datapath):
    #     file_path = os.path.abspath(os.path.join(datapath, file_name))
    #     pid, seq, rep = loadh5(file_path)
    #     samples.append((pid, seq, rep))
    samples=[]
    for file_name in os.listdir(datapath):
        sample_path = os.path.abspath(os.path.join(datapath, file_name))
        samples.append(sample_path)
    
    return samples


def process_samples(data_folder2search, model_name, sub_list, output_folder, frame_method, n_frame, frag_len):
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
            #fragments, frag_contact = fragment_sequence(sequence, np.array([]), fixed_length=224, overlap=30)
            #data_len=len(fragments)
            rep=0
            
            for file_xtc_path in traj_files:
                #data[pid]={'ind2seq':{}, 'ind2map':{}}
                #for i in range(data_len):
                #     data[pid]['ind2map'][i]=[]

                traj = md.load(file_xtc_path, top=file_pdb_path)
                if len(traj)<32:
                    continue
                # cutoff = 7  # in nanometers
                # contact_maps = []
                #traj_stride=1000
                #traj_num=traj.n_frames//traj_stride
                #traj_remain=traj.n_frames%traj_stride
                #traj_list=[]
                #for i in range(traj_num):
                #    distances = md.compute_distances(traj[i*traj_stride:i*traj_stride+traj_stride], 
                #    traj[i*traj_stride:i*traj_stride+traj_stride].topology.select_pairs('name CA', 'name CA'))
                #    contact_maps = vectors_to_contact_maps(distances) # [n, N, N]
                #    traj_list.append(contact_maps)
                #
                #if traj_remain!=0:
                #    distances = md.compute_distances(traj[-traj_remain:], traj[-traj_remain:].topology.select_pairs('name CA', 'name CA'))
                #    contact_maps = vectors_to_contact_maps(distances) # [n, N, N]
                #    traj_list.append(contact_maps)
                #
                #contact_maps = np.concatenate(traj_list, axis=0)
                
                #distances = md.compute_distances(traj, traj.topology.select_pairs('name CA', 'name CA'))
                #contact_maps = vectors_to_contact_maps(distances) # [n, N, N]
                #print(contact_maps.dtype)
                #print(contact_maps.shape)
                #fragments, frag_contacts = fragment_sequence(sequence, contact_maps, fixed_length=224, overlap=30)
                fragments, frag_contacts = fragment_sequence(sequence, traj, fixed_length=frag_len, overlap=30)
                #print(frag_contacts[0].dtype)
                for i in range(len(fragments)):
                    frag_seq=fragments[i]
                    frag_contact=frag_contacts[i]
                    frag_contact = np.expand_dims(frag_contact, axis=-1) # [n, N, N, 1]
                    frag_contact = normalize_contact_map(frag_contact) # [n, N, N, 1]
                    frag_contact = np.repeat(frag_contact, 3, axis=-1) # [n, N, N, 3]
                    #data[pid]['ind2seq'][i]=frag_seq
                    #data[pid]['ind2map'][i]=frag_contact
                #for i in range(len(fragments)):
                    pid_i = pid+"#frag"+str(i)
                    h5id = pid_i+"#rep"+str(rep)
                    #seq_i = data[pid]['ind2seq'][i]
                    seq_i = frag_seq
                    # traj = image_processor(data[pid]['ind2map'][i], return_tensors="pt", do_rescale=False, offset=False, 
                    #                        do_normalize=False, do_resize=True, size=[224, 224]) # [1, 1000, 3, 224, 224]
                    images = image_processor(list(frag_contact), return_tensors="pt", do_rescale=False, offset=False, 
                                           do_normalize=False, do_resize=False) # [1, 1000, 3, 224, 224]
                    images = images['pixel_values']

                    if frame_method=="FFT":
                        images_compressed = recon_FFT_DCT(images, n_frame=32) # [1, 32, 3, 224, 224]
                    elif frame_method=="change_detection":
                        selector = TrajectoryFrameSelector(n_frames=32)
                        images_compressed = selector.select_frames(images, method='change_detection')
                    elif frame_method=="clustering":
                        selector = TrajectoryFrameSelector(n_frames=n_frame)
                        images_compressed = selector.select_frames(images, method='clustering', temporal_order=False)
                    # with torch.no_grad():
                    #         vivit_output = model(images_compressed)
                    # last_hidden_states = vivit_output.last_hidden_state #[1, 3137, 768]
                    # cls_representation = last_hidden_states[0, 0, :] #[1, 768]


                    saveh5((pid_i, seq_i, images_compressed), os.path.join(output_folder, h5id+".h5"))
                    del images, frag_contact
                    torch.cuda.empty_cache()
                    print(get_memory_usage_gb())
                rep+=1
                del traj
                torch.cuda.empty_cache()
                print(get_memory_usage_gb())
    

def fragment_sequence(sequence, traj, fixed_length, overlap):
    fragments = []
    frag_contact = []
    stride = fixed_length - overlap
    start = 0
    seq_len = len(sequence)
    while start < seq_len:
        end = min(start + fixed_length, seq_len)
        fragment = sequence[start:end]
        fragments.append(fragment)  # Add fragment *before* checking length
        atom_indices = traj.topology.select(f"resid {start} to {end-1}")
        sub_traj = traj.atom_slice(atom_indices)
        distances = md.compute_distances(sub_traj, sub_traj.topology.select_pairs('name CA', 'name CA'))
        contact_map = vectors_to_contact_maps(distances) # [n, N, N]
        print(start, end)
        print(contact_map.shape)
        if contact_map.size!=0:
            if end-start<fixed_length:
                background = np.zeros((contact_map.shape[0], 224, 224), dtype=contact_map.dtype)
                #background[:, :end-start, :end-start] = contact_map[:, start:end, start:end]
                background[:, :end-start, :end-start] = contact_map
                frag_contact.append(background)
            else:
                #frag_contact.append(contact_map[:, start:end, start:end])
                frag_contact.append(contact_map)
        if end == seq_len: #check if the end of the sequence has been reached
            break
        start += stride
    return fragments, frag_contact


def saveh5(sample, h5file):
    (pid, seq, pooled) = sample
    # outname = h5file + str()
    os.makedirs(os.path.dirname(h5file), exist_ok=True)
    with h5py.File(h5file, 'w') as f:
    
        # Convert torch tensor to numpy
        grp = f.create_group(pid)
        # grp.attrs['pid'] = pid.encode('utf-8')
        # grp = f.create_group(str(pid))
        grp.create_dataset('pooled', data=pooled, compression='gzip')
        # Save sequence as attribute or dataset (if string, encode first)
        grp.attrs['seq'] = str(seq).encode('utf-8')

def loadh5(h5file):
    with h5py.File(h5file, 'r') as f:
        name = os.path.basename(h5file)
        pid = "#".join(name.split('#')[:2])
        # pid=f[pid].attrs['pid']
        # pid = 'some_pid'
        rep = f[pid]['pooled'][:]
        rep = rep.squeeze(0)
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

def recon_FFT_DCT(traj, n_frame=32): # traj : [1, 1001, 3, 224, 224]
    traj = traj.squeeze(0).float()
    # DCT type-II (cosine transform)
    traj_freq = torch_dct.dct(traj, norm='ortho')
    # Keep first 32 frequency bands (low frequencies)
    traj_freq_low = traj_freq[:n_frame]             # [32, 3, 224, 224]
    # Option A — treat traj_freq_low itself as 32 tokens for a temporal model
    fourier_tokens = traj_freq_low             # already shape [32, 3, 224, 224]
    # Option B — explicitly reconstruct 32 time points in original domain
    # Reconstruct 32 Fourier frames (inverse DCT)
    traj_recon = torch_dct.idct(traj_freq_low, norm='ortho')[:32]   # [32, 3, 224, 224]
    return traj_recon.unsqueeze(0)


class TrajectoryFrameSelector:
    """
    Select most informative frames from protein trajectories.
    
    Input: [T, C, H, W] = [10000, 3, 224, 224]
    Output: [32, 3, 224, 224] or indices of selected frames
    """
    
    def __init__(self, n_frames=32):
        self.n_frames = n_frames
    
    def select_frames(self, video, method='diversity', return_indices=False, temporal_order=True):
        """
        Select informative frames from trajectory.
        
        Parameters:
        -----------
        video : np.ndarray or torch.Tensor
            Shape: [T, C, H, W] = [10000, 3, 224, 224]
        method : str
            'diversity', 'change_detection', 'clustering', 'hybrid', 'uniform'
        return_indices : bool
            If True, return selected frame indices
        temporal_order : bool
            If True, enforce temporal ordering in selection (for diversity/clustering)
            
        Returns:
        --------
        selected_frames : np.ndarray
            Shape: [32, 3, 224, 224]
        indices : list (if return_indices=True)
            Selected frame indices
        """
        video = video.squeeze(0)
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        
        T = video.shape[0]
        print(f"Selecting {self.n_frames} frames from {T} total frames...")
        print(f"Temporal order constraint: {temporal_order}")
        
        if method == 'uniform':
            indices, frames = self._uniform_sampling(video)
        elif method == 'diversity':
            indices, frames = self._diversity_sampling(video, temporal_constraint=temporal_order)
        elif method == 'change_detection':
            indices, frames = self._change_detection(video)
        elif method == 'clustering':
            indices, frames = self._clustering(video, temporal_constraint=temporal_order)
        elif method == 'hybrid':
            indices, frames = self._hybrid_selection(video)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Selected frames at indices: {indices[:5]}...{indices[-5:]}")
        
        if return_indices:
            return torch.from_numpy(frames).unsqueeze(0), indices
        return torch.from_numpy(frames).unsqueeze(0)
    
    def _uniform_sampling(self, video):
        """
        Simple uniform sampling (baseline).
        Always preserves temporal order.
        """
        print("Using uniform sampling...")
        T = video.shape[0]
        indices = np.linspace(0, T-1, self.n_frames, dtype=int).tolist()
        return indices, video[indices]
    
    def _diversity_sampling(self, video, temporal_constraint=False):
        """
        Greedy diversity sampling: select frames that maximize structural diversity.
        Best for: Covering full conformational space
        
        Parameters:
        -----------
        temporal_constraint : bool
            If True, maintain temporal order during selection (slower but ordered)
        """
        print("Using diversity sampling (greedy maximin)...")
        T = video.shape[0]
        
        # Flatten spatial dimensions for distance calculation
        # Use downsampled version for speed
        downsample_factor = 4
        video_small = video[:, :, ::downsample_factor, ::downsample_factor]
        frames_flat = video_small.reshape(T, -1)
        
        # Normalize for better distance metrics
        frames_flat = frames_flat / (np.linalg.norm(frames_flat, axis=1, keepdims=True) + 1e-8)
        
        if temporal_constraint:
            # Temporal-aware diversity: divide timeline into segments
            print("Using temporal constraint (maintains order)...")
            segment_size = T // self.n_frames
            selected = []
            
            for i in range(self.n_frames):
                # Define search window for this segment
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, T)
                
                if i == 0:
                    # First frame: select from first segment
                    selected.append(start_idx)
                else:
                    # Find most diverse frame in this segment
                    segment_indices = range(start_idx, end_idx)
                    segment_frames = frames_flat[segment_indices]
                    selected_frames = frames_flat[selected]
                    
                    # Compute distances to already selected frames
                    dists = np.linalg.norm(
                        segment_frames[:, np.newaxis, :] - selected_frames[np.newaxis, :, :],
                        axis=2
                    )
                    min_dists = dists.min(axis=1)
                    
                    # Select frame with maximum minimum distance
                    best_in_segment = segment_indices[min_dists.argmax()]
                    selected.append(best_in_segment)
            
            return selected, video[selected]
        
        else:
            # Original global diversity sampling
            selected = [0]  # Always include first frame
            
            print("Computing pairwise distances (this may take a moment)...")
            # Compute distances in batches to save memory
            batch_size = 1000
            
            for i in tqdm(range(1, self.n_frames), desc="Selecting frames"):
                max_min_dist = -1
                best_idx = -1
                
                for batch_start in range(0, T, batch_size):
                    batch_end = min(batch_start + batch_size, T)
                    batch_indices = range(batch_start, batch_end)
                    
                    # Skip already selected
                    batch_indices = [idx for idx in batch_indices if idx not in selected]
                    if not batch_indices:
                        continue
                    
                    # Compute distances to all selected frames
                    batch_frames = frames_flat[batch_indices]
                    selected_frames = frames_flat[selected]
                    
                    # Distance matrix: [batch_size, n_selected]
                    dists = np.linalg.norm(
                        batch_frames[:, np.newaxis, :] - selected_frames[np.newaxis, :, :],
                        axis=2
                    )
                    
                    # Minimum distance to any selected frame
                    min_dists = dists.min(axis=1)
                    
                    # Find frame with maximum minimum distance
                    batch_best_idx = min_dists.argmax()
                    if min_dists[batch_best_idx] > max_min_dist:
                        max_min_dist = min_dists[batch_best_idx]
                        best_idx = batch_indices[batch_best_idx]
                
                if best_idx != -1:
                    selected.append(best_idx)
            
            selected = sorted(selected)
            return selected, video[selected]
    
    def _change_detection(self, video):
        """
        Select frames with largest conformational changes.
        Best for: Capturing transition states and events
        """
        print("Using change detection...")
        T = video.shape[0]
        
        # Compute frame-to-frame differences
        print("Computing frame differences...")
        diffs = np.zeros(T - 1)
        for i in tqdm(range(T - 1), desc="Computing differences"):
            diff = video[i+1] - video[i]
            diffs[i] = np.linalg.norm(diff)
        
        # Normalize differences
        diffs = (diffs - diffs.mean()) / (diffs.std() + 1e-8)
        
        # Find peaks (large changes)
        min_distance = T // (self.n_frames * 3)  # Ensure frames are spread out
        peaks, properties = find_peaks(diffs, distance=min_distance, prominence=0.5)
        
        print(f"Found {len(peaks)} significant change peaks")
        
        # Combine peaks with uniform sampling
        if len(peaks) >= self.n_frames:
            # Too many peaks - select top ones
            top_peaks = peaks[np.argsort(diffs[peaks])[-self.n_frames:]]
            selected = sorted(top_peaks)
        else:
            # Not enough peaks - add uniform samples
            n_uniform = self.n_frames - len(peaks)
            uniform = np.linspace(0, T-1, n_uniform, dtype=int)
            
            # Remove uniform samples too close to peaks
            if len(peaks) > 0:
                uniform_filtered = []
                for u in uniform:
                    if np.min(np.abs(peaks - u)) > min_distance:
                        uniform_filtered.append(u)
                uniform = np.array(uniform_filtered)
            
            # Combine and fill if needed
            selected = np.concatenate([peaks, uniform])
            if len(selected) < self.n_frames:
                # Still need more - add evenly spaced
                remaining = self.n_frames - len(selected)
                extra = np.linspace(0, T-1, remaining, dtype=int)
                selected = np.concatenate([selected, extra])
            
            selected = sorted(np.unique(selected))[:self.n_frames]
        
        return selected, video[selected]
    
    def _clustering(self, video, temporal_constraint=False):
        """
        K-means clustering in feature space.
        Best for: Representative conformations
        
        Parameters:
        -----------
        temporal_constraint : bool
            If True, cluster within temporal windows
        """
        print("Using clustering (MiniBatch K-means)...")
        T = video.shape[0]
        
        # Downsample for speed
        downsample_factor = 4
        video_small = video[:, :, ::downsample_factor, ::downsample_factor]
        frames_flat = video_small.reshape(T, -1)
        
        if temporal_constraint:
            # Temporal-aware clustering: cluster within windows
            print("Using temporal constraint (maintains order)...")
            segment_size = T // self.n_frames
            selected = []
            
            for i in range(self.n_frames):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, T)
                
                # Select middle frame of segment (simple approach)
                # Or could do mini-clustering within segment
                mid_idx = (start_idx + end_idx) // 2
                selected.append(mid_idx)
            
            return selected, video[selected]
        
        else:
            # Use MiniBatchKMeans for large datasets
            print(f"Clustering {T} frames into {self.n_frames} clusters...")
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_frames,
                random_state=42,
                batch_size=1000,
                max_iter=100,
                verbose=1
            )
            labels = kmeans.fit_predict(frames_flat)
            
            # Select frame closest to each cluster center
            print("Selecting representative frames from each cluster...")
            selected = []
            for i in range(self.n_frames):
                cluster_frames = np.where(labels == i)[0]
                if len(cluster_frames) > 0:
                    # Find frame closest to cluster center
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(
                        frames_flat[cluster_frames] - center, axis=1
                    )
                    selected.append(cluster_frames[distances.argmin()])
                else:
                    # Empty cluster - add a uniform sample
                    selected.append(int(i * T / self.n_frames))
            
            selected = sorted(selected)
            return selected, video[selected]
    
    def _hybrid_selection(self, video):
        """
        Hybrid approach: combine change detection and diversity.
        Best for: Balanced representation
        """
        print("Using hybrid approach (change detection + diversity)...")
        
        # Get half frames from change detection
        n_change = self.n_frames // 2
        n_diverse = self.n_frames - n_change
        
        temp_selector1 = TrajectoryFrameSelector(n_frames=n_change)
        change_indices, _ = temp_selector1._change_detection(video)
        
        # Get remaining frames from diversity sampling on non-selected frames
        remaining_indices = [i for i in range(len(video)) if i not in change_indices]
        remaining_video = video[remaining_indices]
        
        temp_selector2 = TrajectoryFrameSelector(n_frames=n_diverse)
        diverse_rel_indices, _ = temp_selector2._diversity_sampling(remaining_video)
        diverse_indices = [remaining_indices[i] for i in diverse_rel_indices]
        
        selected = sorted(change_indices + diverse_indices)
        return selected, video[selected]
    
    def visualize_selection(self, video, selected_indices, save_path=None):
        """
        Visualize selected frames on timeline.
        """
        T = video.shape[0]
        
        # Compute frame differences for context
        diffs = np.zeros(T - 1)
        for i in range(T - 1):
            diff = video[i+1] - video[i]
            diffs[i] = np.linalg.norm(diff.flatten())
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Top: Timeline with selected frames
        ax1 = axes[0]
        ax1.plot(range(T-1), diffs, 'b-', alpha=0.5, linewidth=0.5, label='Frame difference')
        ax1.scatter(selected_indices, [diffs[min(i, T-2)] for i in selected_indices],
                   c='red', s=100, zorder=5, label=f'Selected ({len(selected_indices)} frames)')
        ax1.set_xlabel('Frame Index', fontsize=12)
        ax1.set_ylabel('Frame Difference Magnitude', fontsize=12)
        ax1.set_title('Selected Frames on Trajectory Timeline', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Bottom: Selected frame thumbnails
        ax2 = axes[1]
        n_show = min(16, len(selected_indices))
        show_indices = np.linspace(0, len(selected_indices)-1, n_show, dtype=int)
        
        thumbnail_strip = []
        for idx in show_indices:
            frame_idx = selected_indices[idx]
            frame = video[frame_idx].transpose(1, 2, 0)  # CHW -> HWC
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
            thumbnail_strip.append(frame)
        
        thumbnail_strip = np.concatenate(thumbnail_strip, axis=1)
        ax2.imshow(thumbnail_strip)
        ax2.set_xticks([(i + 0.5) * 224 for i in range(n_show)])
        ax2.set_xticklabels([f"t={selected_indices[i]}" for i in show_indices], rotation=45)
        ax2.set_yticks([])
        ax2.set_title(f'Sample of Selected Frames (showing {n_show}/{len(selected_indices)})', 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--data_folder2search", help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/')
    parser.add_argument("--model_name", default='google/vivit-b-16x2-kinetics400', help='xx')
    parser.add_argument("--list_num", type=int, default=0, help="xx")
    parser.add_argument("--output_folder", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data', help="xx")
    parser.add_argument("--split_num", type=int, default=6, help="xx")
    parser.add_argument("--frame_select_method", default='clustering', help="xx")
    parser.add_argument("--frame_num", type=int, default=32, help="xx")
    parser.add_argument("--frag_len", type=int, default=224, help="xx")

    
    args_main = parser.parse_args()
    lists = split_subfolders_into_n_lists(args_main.data_folder2search, n=args_main.split_num)
    process_samples(args_main.data_folder2search, args_main.model_name,
                     sub_list=lists[args_main.list_num], output_folder=args_main.output_folder, frame_method=args_main.frame_select_method, n_frame=args_main.frame_num,
                     frag_len=args_main.frag_len)
