import glob
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import tqdm
import os
from torch.utils.data import DataLoader, Dataset
from .utils import load_h5_file
from torch_geometric.data import Batch, Data
import time
from .rotary_embedding import RotaryEmbedding

"""Yichuan: New imports"""
import random
import csv

def custom_collate(one_batch):
    # Unpack the batch
    torch_geometric_feature = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data

    # Create a Batch object
    torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    raw_seqs = [item[1] for item in one_batch]
    plddt_scores = [item[2] for item in one_batch]
    pids = [item[3] for item in one_batch]
    plddt_scores = torch.cat(plddt_scores, dim=0)
    batched_data = {'graph': torch_geometric_batch, 'seq': raw_seqs, 'plddt': plddt_scores, 'pid': pids}
    return batched_data


def custom_collate_contrastive_learning(one_batch):
    """
    Yichuan:
    This is an upgraded version of the custom_collate() function.
    It automatically detects whether it is in contrastive learning mode.
    If not, it functions identically to the original custom_collate, simply aggregating the batch data without any modifications.
    If it is in contrastive learning mode, the function constructs a new batch that includes not only the anchor samples,
    but also the corresponding positive and negative samples directly appended to the batch.
    """

    if len(one_batch[0]) == 4:  # custom_collate()
        # Unpack the batch
        torch_geometric_feature = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data

        # Create a Batch object
        torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
        raw_seqs = [item[1] for item in one_batch]
        plddt_scores = [item[2] for item in one_batch]
        pids = [item[3] for item in one_batch]

        plddt_scores = torch.cat(plddt_scores, dim=0)
        batched_data = {'graph': torch_geometric_batch, 'seq': raw_seqs, 'plddt': plddt_scores, 'pid': pids}
        return batched_data
    else:
        # Initialize lists to hold batch data
        all_features = []
        all_seqs = []
        all_plddt_scores = []
        all_pids = []

        # Iterate through each item in the batch
        for item in one_batch:
            # Unpack the data into anchor samples and potential contrastive samples (positive and negative)
            anchor_feature, anchor_seq, anchor_plddt_scores, anchor_pid, pos_samples, neg_samples = item

            # Add anchor sample data to the batch
            all_features.append(anchor_feature)
            all_seqs.append(anchor_seq)
            all_plddt_scores.append(anchor_plddt_scores)
            all_pids.append(anchor_pid)

            # If in contrastive learning mode and both positive and negative samples are not None
            if pos_samples is not None and neg_samples is not None:
                # Add positive sample data to the batch
                for pos in pos_samples:
                    all_features.append(pos[0])
                    all_seqs.append(pos[1])
                    all_plddt_scores.append(pos[2])
                    all_pids.append(pos[3])

                # Add negative sample data to the batch
                for neg in neg_samples:
                    all_features.append(neg[0])
                    all_seqs.append(neg[1])
                    all_plddt_scores.append(neg[2])
                    all_pids.append(neg[3])
            else:
                raise ValueError("Missing positive or negative samples for contrastive learning.")

        # Combine all graph features into a single Batch object
        torch_geometric_batch = Batch.from_data_list(all_features)
        # Concatenate all plddt scores into a single Tensor
        all_plddt_scores = torch.cat(all_plddt_scores, dim=0)

        # Create a dictionary containing all combined batch data
        batched_data = {
            'graph': torch_geometric_batch,
            'seq': all_seqs,
            'plddt': all_plddt_scores,
            'pid': all_pids
        }
        # raise ValueError(batched_data['pid'])
        return batched_data


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(d, d_min=0., d_max=20., d_count=16, device='cpu'):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `d` along a new axis=-1.
    That is, if `d` has shape [...dims], then the returned tensor will have
    shape [...dims, d_count].
    """
    d_mu = torch.linspace(d_min, d_max, d_count,
                          # device=device
                          )
    d_mu = d_mu.view([1, -1])
    d_sigma = (d_max - d_min) / d_count
    d_sigma = torch.tensor(d_sigma,
                           # device=device
                           )  # Convert d_sigma to a tensor
    d_expand = torch.unsqueeze(d, -1)

    RBF = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
    return RBF


def get_pdb_ec_dict_and_ec_list(filename):
    pdb_ec_dict = {} #{pdb_chain:ec_numbers} 
    unique_ec_numbers = set()
    #ec_id = {}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)
        next(reader, None)
        next(reader, None)
        for row in reader:
            pdb_chain = row[0]
            ec_numbers = row[1].split(',')
            pdb_ec_dict[pdb_chain] = ec_numbers
            unique_ec_numbers.update(ec_numbers)
            
            #for ec in ec_numbers:
            #    if ec not in ec_id.keys():
            #        ec_id[ec] = set()
            #        ec_id[ec].add(pdb_chain)
            #    else:
            #        ec_id[ec].add(pdb_chain)
    
    label_list = list(unique_ec_numbers)
    return pdb_ec_dict, label_list


def get_pdb_go_dict_and_go_list(filename, go_term):
    pdb_go_dict = {}
    #go_id = {}
    unique_go_numbers = set()
    index = 0
    if go_term == 'molecular_function':
        index = 1
    elif go_term == 'biological_process':
        index = 2
    elif go_term == 'cellular_component':
        index = 3
    else:
        raise ValueError(f'Invalid GO term: {go_term}')
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        next(reader, None)
        for row in reader:
            _row = [element for element in row if element]
            row = _row  # This is to remove the '' empty elements,
            # this bug is caused by the poor written GO annotation file
            # check 5KYH-A, row = ['5KYH-A', '', '', 'GO:0016021,GO:0031224']
            pdb_chain = row[0]
            go_numbers = row[index].split(',')
            pdb_go_dict[pdb_chain] = go_numbers
            unique_go_numbers.update(go_numbers)
            #for go in gp_numbers:
            # if go not in go_id.keys():
            #    go_id[go] = set()
            #    go_id[go].add(pdb_chain)
            # else:
            #    go_id[go].add(pdb_chain)
    
    label_list = list(unique_go_numbers)
    return pdb_go_dict, label_list


class ProteinGraphDataset(Dataset):

    def __init__(self, data_path,
                 max_length=512,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cuda",
                 seq_mode="embedding", use_rotary_embeddings=False, rotary_mode=1,
                 use_foldseek=False, use_foldseek_vector=False,
                 # new parameters below
                 apply_contrastive_learning=False,
                 n_pos=0,
                 n_neg=0,
                 contrastive_sample_granularity=4,
                 dataset_name=None,
                 pdb_ec_dict=None,
                 contact_cutoff = None,
                 ):
        
        """
        pdb_ec_dict is provided for training and evaluation, if not provided, 
        create dataloader for novel test dataset and set self.dataset_name to "testdataset"
        """
        """Yichuan: Pay attention to the settings for contrastive learning parameters."""
        super(ProteinGraphDataset, self).__init__()
        """"""
        self.contact_cutoff = contact_cutoff
        self.max_length = max_length
        if pdb_ec_dict is not None: #the ec number is provided, for train, val and test samples that have annotations
            self.pdb_ec_dict = pdb_ec_dict
            if dataset_name == 'CATH':
                self.dataset_name = 'CATH'
                self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))
            elif dataset_name == 'EC' or dataset_name == 'GO':
                self.dataset_name = dataset_name
                self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))
                if len(self.h5_samples) == 0:
                    raise ValueError('Did not retrieve any filenames. Please verify that the data path is correct.')
                new_h5_samples = []
                for file_path in self.h5_samples:
                    file_name = os.path.basename(file_path)[0:-3]
                    if file_name in ['6GMH-V', '6GMH-U', '6GML-X']:
                        continue
                    if file_name in pdb_ec_dict:
                        ec_list = pdb_ec_dict[file_name]
                        for ec in ec_list:
                            new_file_name = f"{os.path.splitext(file_name)[0]}_{ec}.h5"
                            new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
                            new_h5_samples.append(new_file_path)
                    else:
                        raise ValueError(f"No corresponding {dataset_name} list found for file: {file_name}")
                self.h5_samples = new_h5_samples
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
        else: #for novel test data, no ec number is provided
            self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))
            if len(self.h5_samples) == 0:
                raise ValueError('Did not retrieve any filenames. Please verify that the data path is correct.')
            
            self.dataset_name = "testdataset"
        
        """"""
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.seq_mode = seq_mode
        # self.node_counts = [len(e['seq']) for e in data_list]
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rotary_mode = rotary_mode
        self.use_foldseek = use_foldseek
        self.use_foldseek_vector = use_foldseek_vector
        if self.use_rotary_embeddings:
            if self.rotary_mode == 3:
                self.rot_emb = RotaryEmbedding(dim=8)  # must be 5
            else:
                self.rot_emb = RotaryEmbedding(dim=2)  # must be 2

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

        self.letter_to_PhysicsPCA = {"C": [-0.132, 0.174, 0.070, 0.565, -0.374],
                                     "D": [0.303, -0.057, -0.014, 0.225, 0.156],
                                     "S": [0.199, 0.238, -0.015, -0.068, -0.196],
                                     "Q": [0.149, -0.184, -0.030, 0.035, -0.112],
                                     "K": [0.243, -0.339, -0.044, -0.325, -0.027],
                                     "I": [-0.353, 0.071, -0.088, -0.195, -0.107],
                                     "P": [0.173, 0.286, 0.407, -0.215, 0.384],
                                     "T": [0.068, 0.147, -0.015, -0.132, -0.274],
                                     "F": [-0.329, -0.023, 0.072, -0.002, 0.208],
                                     "A": [0.008, 0.134, -0.475, -0.039, 0.181],
                                     "G": [0.218, 0.562, -0.024, 0.018, 0.106],
                                     "H": [0.023, -0.177, 0.041, 0.280, -0.021],
                                     "E": [0.221, -0.280, -0.315, 0.157, 0.303],
                                     "L": [-0.267, 0.018, -0.265, -0.274, 0.206],
                                     "R": [0.171, -0.361, 0.107, -0.258, -0.364],
                                     "W": [-0.296, -0.186, 0.389, 0.083, 0.297],
                                     "V": [-0.274, 0.136, -0.187, -0.196, -0.299],
                                     "Y": [-0.141, -0.057, 0.425, -0.096, -0.091],
                                     "N": [0.255, 0.038, 0.117, 0.118, -0.055],
                                     "M": [-0.239, -0.141, -0.155, 0.321, 0.077]}

        # https://www.pnas.org/doi/full/10.1073/pnas.0408677102#sec-4
        self.letter_to_Atchleyfactor = {
            "A": [-0.591, -1.302, -0.733, 1.570, -0.146],
            "C": [-1.343, 0.465, -0.862, -1.020, -0.255],
            "D": [1.050, 0.302, -3.656, -0.259, -3.242],
            "E": [1.357, -1.453, 1.477, 0.113, -0.837],
            "F": [-1.006, -0.590, 1.891, -0.397, 0.412],
            "G": [-0.384, 1.652, 1.330, 1.045, 2.064],
            "H": [0.336, -0.417, -1.673, -1.474, -0.078],
            "I": [-1.239, -0.547, 2.131, 0.393, 0.816],
            "K": [1.831, -0.561, 0.533, -0.277, 1.648],
            "L": [-1.019, -0.987, -1.505, 1.266, -0.912],
            "M": [-0.663, -1.524, 2.219, -1.005, 1.212],
            "N": [0.945, 0.828, 1.299, -0.169, 0.933],
            "P": [0.189, 2.081, -1.628, 0.421, -1.392],
            "Q": [0.931, -0.179, -3.005, -0.503, -1.853],
            "R": [1.538, -0.055, 1.502, 0.440, 2.897],
            "S": [-0.228, 1.399, -4.760, 0.670, -2.647],
            "T": [-0.032, 0.326, 2.213, 0.908, 1.313],
            "V": [-1.337, -0.279, -0.544, 1.242, -1.262],
            "W": [-0.595, 0.009, 0.672, -2.128, -0.184],
            "Y": [0.260, 0.830, 3.097, -0.838, 1.512]}

        """Yichuan: Pay attention to the settings for contrastive learning parameters."""
        self.apply_contrastive_learning = apply_contrastive_learning
        if apply_contrastive_learning:
            self.n_pos = n_pos
            self.n_neg = n_neg
            self.contrastive_sample_granularity = contrastive_sample_granularity

    def __len__(self):
        return len(self.h5_samples)

    def get_pos_samples(self, i, n_pos, granularity=4):
        anchor_path = self.h5_samples[i]
        anchor_basename = os.path.basename(anchor_path)
        anchor_label = anchor_basename.split('.h5')[0].split("_")[1]
        #Duolin if anchor has - for EC, set granularity to the index of "-".
        if "-" in anchor_label:
            index = np.where(np.asarray(anchor_label.split("."))== '-')[0][0]
            granularity = index
        
        anchor_label = '_' + anchor_label
        
        if granularity in (1, 2, 3, 4):
            parts = anchor_label.split('.')
            anchor_label = '.'.join(parts[:granularity])
        else:
            # The granularity is not supported
            raise ValueError(f"Unsupported granularity provided: {granularity}")
        
        # Find all samples that match the anchor label but are not the anchor itself
        matching_samples = [s for s in self.h5_samples if
                            anchor_label in '_' + os.path.basename(s).split('.h5')[0].split("_")[1] and s != anchor_path]
        # Check if there are any matching samples available
        if len(matching_samples) == 0:
            # As a fallback, use the anchor itself if no other samples are available
            selected_paths = [anchor_path] * n_pos
            print(f"Warning: No positive samples found other than the anchor. Anchor duplicated.")
        elif len(matching_samples) >= n_pos:
            # If there are enough unique matching samples, randomly select the required number
            selected_paths = random.sample(matching_samples, n_pos)
        else:
            # Not enough matching samples to meet the requirement, use random.choices to allow duplicates
            selected_paths = random.choices(matching_samples, k=n_pos)
            print(f"Warning: Insufficient unique positive samples ({len(matching_samples)}) found for label '{anchor_label}' in {anchor_path}. Duplicates were used.")

        # Process each selected path to format its data as required
        selected_samples = []
        for path in selected_paths:
            """"""
            if self.dataset_name == 'CATH':
                sequence,coord,plddt_scores = load_h5_file(path)
            elif self.dataset_name == 'EC' or self.dataset_name == 'GO':
                base_name = os.path.basename(path)
                dir_name = os.path.dirname(path)
                # base_name_without_ec = base_name.split('_')[0] + '.h5'
                if '_unique' in base_name:
                    base_name_without_ec = base_name.split('_')[0] + '_unique' + '.h5'
                else:
                    base_name_without_ec = base_name.split('_')[0] + '.h5'
                real_path = os.path.join(dir_name, base_name_without_ec)
                sequence,coord,plddt_scores = load_h5_file(real_path)
            """"""
            pid = os.path.basename(path).split('.h5')[0]
            coord = coord[:self.max_length]
            sequence = sequence[:self.max_length]
            sample_dict = {
                'name': pid,
                'coords': coord.tolist(),
                'seq': sequence.decode('utf-8')
            }
            feature = self._featurize_as_graph(sample_dict)
            plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
            raw_seqs = sequence.decode('utf-8')
            # Append the formatted sample to the selected_samples list
            selected_samples.append([feature, raw_seqs, plddt_scores, pid])

        return selected_samples

    def get_neg_samples(self, i, n_neg, granularity=4):
        anchor_path = self.h5_samples[i]
        anchor_basename = os.path.basename(anchor_path)
        anchor_label = anchor_basename.split('.h5')[0].split("_")[1]
        #Duolin if anchor has - for EC, set granularity to the index of "-".
        if "-" in anchor_label:
            index = np.where(np.asarray(anchor_label.split("."))== '-')[0][0]
            if granularity > index:
               granularity = index
        
        anchor_label = '_' + anchor_label

        if granularity in (1, 2, 3, 4):
            parts = anchor_label.split('.')
            anchor_label = '.'.join(parts[:granularity])
        else:
            # Handling other modes or raising an error if the mode is not supported
            raise ValueError(f"Unsupported granularity provided: {granularity}")

        # Find all samples that do NOT match the anchor label
        matching_samples = [s for s in self.h5_samples if
                            anchor_label not in '_' + os.path.basename(s).split('.h5')[0].split("_")[1]]
        # If there are enough unique neg samples (must be true), randomly select the required number
        selected_paths = random.sample(matching_samples, n_neg)

        # Process each selected path to format its data as required
        selected_samples = []
        for path in selected_paths:
            """"""
            if self.dataset_name == 'CATH':
                sequence,coord,plddt_scores = load_h5_file(path)
            elif self.dataset_name == 'EC' or self.dataset_name == 'GO':
                base_name = os.path.basename(path)
                dir_name = os.path.dirname(path)
                # base_name_without_ec = base_name.split('_')[0] + '.h5'
                if '_unique' in base_name:
                    base_name_without_ec = base_name.split('_')[0] + '_unique' + '.h5'
                else:
                    base_name_without_ec = base_name.split('_')[0] + '.h5'
                real_path = os.path.join(dir_name, base_name_without_ec)
                sequence,coord,plddt_scores = load_h5_file(real_path)
            """"""
            pid = os.path.basename(path).split('.h5')[0]
            coord = coord[:self.max_length]
            sequence = sequence[:self.max_length]
            sample_dict = {
                'name': pid,
                'coords': coord.tolist(),
                'seq': sequence.decode('utf-8')
            }
            feature = self._featurize_as_graph(sample_dict)
            plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
            raw_seqs = sequence.decode('utf-8')
            # Append the formatted sample to the selected_samples list
            selected_samples.append([feature, raw_seqs, plddt_scores, pid])

        return selected_samples

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        """"""
        if self.dataset_name == 'CATH':
            sequence,coord,plddt_scores  = load_h5_file(sample_path)
        elif self.dataset_name == 'EC' or self.dataset_name == 'GO':
            base_name = os.path.basename(sample_path)
            dir_name = os.path.dirname(sample_path)
            # base_name_without_ec = base_name.split('_')[0] + '.h5'
            if '_unique' in base_name:
                base_name_without_ec = base_name.split('_')[0] + '_unique' + '.h5'
            else:
                base_name_without_ec = base_name.split('_')[0] + '.h5'
            real_sample_path = os.path.join(dir_name, base_name_without_ec)
            sequence,coord,plddt_scores  = load_h5_file(real_sample_path)
        else:
            sequence,coord,plddt_scores  = load_h5_file(sample_path)
        """"""
        coord = coord[:self.max_length]
        sequence = sequence[:self.max_length]
        pid = os.path.basename(sample_path).split('.h5')[0]

        sample_dict = {'name': pid,
                       'coords': coord.tolist(),
                       'seq': sequence.decode('utf-8')}
        feature = self._featurize_as_graph(sample_dict)
        plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
        raw_seqs = sequence.decode('utf-8')

        """
        Yichuan: 
        if apply contrastive learning
        positive samples and negative samples not None
        """
        if self.apply_contrastive_learning:
            pos_samples = self.get_pos_samples(i, self.n_pos, 4)
            neg_samples = self.get_neg_samples(i, self.n_neg, self.contrastive_sample_granularity)
            """print begin"""
            ##parts = pid.split('.')
            ##anchor_pid = '.'.join(parts[:self.contrastive_sample_granularity])
            #anchor_pid = pid
            #print('anc:', anchor_pid, flush=True)
            #print('pos: ', end='', flush=True)
            #for sample in pos_samples:
            #    print(sample[3], end=', ', flush=True)
            #print()
            #print('neg: ', end='')
            #for sample in neg_samples:
            #    print(sample[3], end=', ', flush=True)
            #print('\n')
            """print end"""
            return [feature, raw_seqs, plddt_scores, pid, pos_samples, neg_samples]
        else:
            return [feature, raw_seqs, plddt_scores, pid]

    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            # x=N, C-alpha, C, and O atoms
            coords = torch.as_tensor(protein['coords'],
                                     # device=self.device,
                                     dtype=torch.float32)
            # print(self.seq_mode)
            if self.seq_mode == "PhysicsPCA":
                seq = torch.as_tensor([self.letter_to_PhysicsPCA[a] for a in protein['seq']],
                                      # device=self.device,
                                      dtype=torch.long)
            elif self.seq_mode == "Atchleyfactor":
                seq = torch.as_tensor([self.letter_to_Atchleyfactor[a] for a in protein['seq']],
                                      # device=self.device,
                                      dtype=torch.long)
            elif self.seq_mode == "embedding":
                seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                      # device=self.device,
                                      dtype=torch.long)
            elif self.seq_mode == "ESM2":
                seq = None
            elif self.seq_mode == "noseq":
                seq = None
            if len(coords.shape) < 3:
                raise ValueError(name, coords.shape)
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]] 
            ca_distances_topk = E_vectors.norm(dim=-1)
            #求topk 有问题，不是所有的topk 都应该保留，比如有的topk的邻居其实距离很远，就不应该看它了。
            if self.contact_cutoff is not None:
                #print(ca_distances_topk)
                #print(ca_distances_topk[:3,:3])
                mask = ca_distances_topk <= self.contact_cutoff
                filtered_edge_index = edge_index[:,mask]
                # First, get the nearest neighbors to ensure connectivity
                nearest_edge_index = torch_cluster.knn_graph(X_ca, k=1)
                final_edge_index = torch.cat([nearest_edge_index, filtered_edge_index], dim=1).unique(dim=1)
                edge_index = final_edge_index
                E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]] 
                ca_distances_topk = E_vectors.norm(dim=-1)
            
            if self.use_rotary_embeddings:
                if self.rotary_mode == 1:  # first mode
                    d1 = edge_index[0] - edge_index[1]
                    d2 = edge_index[1] - edge_index[0]
                    d = torch.cat((d1.unsqueeze(-1), d2.unsqueeze(-1)), dim=-1)  # [len,2]
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
                if self.rotary_mode == 2:
                    d = edge_index.transpose(0, 1)
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
                if self.rotary_mode == 3:
                    d = edge_index.transpose(0, 1)  # [len,2]
                    d = torch.cat((d, E_vectors, -1 * E_vectors), dim=-1)  # [len,2+3]
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
            else:
                pos_embeddings = self._positional_embeddings(edge_index)

            rbf = _rbf(ca_distances_topk, d_count=self.num_rbf,
                       # device=self.device
                       )

            dihedrals = self._dihedrals(coords)  # only this one used O
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            if self.use_foldseek or self.use_foldseek_vector:
                foldseek_features = self._foldseek(X_ca)

            if self.use_foldseek:
                node_s = torch.cat([dihedrals, foldseek_features[0]], dim=-1)  # add additional 10 features
            else:
                node_s = dihedrals

            if self.use_foldseek_vector:
                node_v = torch.cat([orientations, sidechains.unsqueeze(-2), foldseek_features[1]],
                                   dim=-2)  # add additional 18 features
            else:
                node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)

            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2) 
            #自己对自己做norm与其他的边没有关系。怎么能是vector呢？这里不应该是scalar么？

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))

        data = Data(x=X_ca, seq=seq, name=name,
                    node_s=node_s, node_v=node_v,
                    edge_s=edge_s, edge_v=edge_v,
                    edge_index=edge_index, mask=mask)
        return data

    @staticmethod
    def _dihedrals(x, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        x = torch.reshape(x[:, :3], [3 * x.shape[0], 3])
        dx = x[1:] - x[:-1]
        U = _normalize(dx, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.linalg.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=(2, 1000)):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                         # device=self.device
                         )
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    @staticmethod
    def _orientations(x):
        forward = _normalize(x[1:] - x[:-1])
        backward = _normalize(x[:-1] - x[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    @staticmethod
    def _sidechains(x):
        n, origin, c = x[:, 0], x[:, 1], x[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.linalg.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        # The specific weights used in the linear combination (math.sqrt(1 / 3) and math.sqrt(2 / 3)) are derived from the idealized tetrahedral geometry of the sidechain atoms around the C-alpha atom. However, in practice, these weights may be adjusted or learned from data to better capture the actual sidechain geometries observed in protein structures.
        return vec

    @staticmethod
    def _foldseek(x):
        # From Fast and accurate protein structure search with Foldseek
        # x is X_ca coordinates
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        j, i = torch_cluster.knn_graph(x, k=1)
        mask = torch.zeros_like(i, dtype=torch.bool)
        first_unique = {}
        for index, value in enumerate(i):
            if value.item() not in first_unique:
                first_unique[value.item()] = index
                mask[index] = True
            else:
                mask[index] = False

        i, j = i[mask], j[mask]
        pad_x = F.pad(x, [0, 0, 1, 1])  # padding to the first and last
        j = j + 1
        i = i + 1
        d = torch.norm(pad_x[i] - pad_x[j], dim=-1, keepdim=True)
        abs_diff = torch.abs(i - j)
        sign = torch.sign(i - j)
        f1 = (sign * torch.minimum(abs_diff, torch.tensor(4))).unsqueeze(-1)
        f2 = (sign * torch.log(abs_diff + 1)).unsqueeze(-1)

        u1 = _normalize(pad_x[i] - pad_x[i - 1])
        u2 = _normalize(pad_x[i + 1] - pad_x[i])
        u3 = _normalize(pad_x[j] - pad_x[j - 1])
        u4 = _normalize(pad_x[j + 1] - pad_x[j])
        u5 = _normalize(pad_x[j] - pad_x[i])
        cos_12 = torch.sum(u1 * u2, dim=-1, keepdim=True)
        cos_34 = torch.sum(u3 * u4, dim=-1, keepdim=True)
        cos_15 = torch.sum(u1 * u5, dim=-1, keepdim=True)
        cos_35 = torch.sum(u3 * u5, dim=-1, keepdim=True)
        cos_14 = torch.sum(u1 * u4, dim=-1, keepdim=True)
        cos_23 = torch.sum(u2 * u3, dim=-1, keepdim=True)
        cos_13 = torch.sum(u1 * u3, dim=-1, keepdim=True)
        node_s_features = torch.cat([cos_12, cos_34, cos_15, cos_35, cos_14, cos_23, cos_13, d, f1, f2], dim=-1)
        node_v_features = torch.cat([u1.unsqueeze(-2), u2.unsqueeze(-2), u3.unsqueeze(-2),
                                     u4.unsqueeze(-2), u5.unsqueeze(-2), (pad_x[i] - pad_x[j]).unsqueeze(-2)],
                                    dim=-2)  # add 6 additional features
        return node_s_features, node_v_features


def prepare_dataloaders(logging, accelerator, configs):
    if accelerator.is_main_process:
        logging.info(f"train directory: {configs.train_settings.data_path}")
        logging.info(f"valid directory: {configs.valid_settings.data_path}")

    if hasattr(configs.model.struct_encoder, "use_seq") and configs.model.struct_encoder.use_seq.enable:
        seq_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
    else:
        seq_mode = "noseq"

    train_dataset = ProteinGraphDataset(configs.train_settings.data_path, 
                                        max_length = configs.model.esm_encoder.max_length,seq_mode=seq_mode,
                                        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                        rotary_mode=configs.model.struct_encoder.rotary_mode,
                                        use_foldseek=configs.model.struct_encoder.use_foldseek,
                                        use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
                                        top_k=configs.model.struct_encoder.top_k,
                                        num_rbf=configs.model.struct_encoder.num_rbf,
                                        num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings
                                        )
    val_dataset = ProteinGraphDataset(configs.valid_settings.data_path,
                                      max_length = configs.model.esm_encoder.max_length,seq_mode=seq_mode,
                                      use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                      rotary_mode=configs.model.struct_encoder.rotary_mode,
                                      use_foldseek=configs.model.struct_encoder.use_foldseek,
                                      use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
                                      top_k=configs.model.struct_encoder.top_k,
                                      num_rbf=configs.model.struct_encoder.num_rbf,
                                      num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings
                                      )
    
    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                              shuffle=configs.train_settings.shuffle,
                              num_workers=configs.train_settings.num_workers,
                              multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=True,
                              collate_fn=custom_collate)  # Yichuan
    val_loader = DataLoader(val_dataset, batch_size=configs.valid_settings.batch_size,
                            num_workers=configs.valid_settings.num_workers,
                            shuffle=False,
                            multiprocessing_context='spawn' if configs.valid_settings.num_workers > 0 else None,
                            pin_memory=True,
                            collate_fn=custom_collate)
    
    
    return train_loader, val_loader


def prepare_dataloaders_contrastive_learning(logging, accelerator, configs):
    if accelerator.is_main_process:
        logging.info(f"train directory: {configs.train_settings.data_path}")
        logging.info(f"valid directory: {configs.valid_settings.data_path}")

    if hasattr(configs.model.struct_encoder, "use_seq") and configs.model.struct_encoder.use_seq.enable:
        seq_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
    else:
        seq_mode = "noseq"

    # pdb_ec_dict, label_list = get_pdb_ec_dict_and_ec_list(configs.dataset.annotation_path)
    if configs.dataset.name == 'EC':
        pdb_ec_dict,_ = get_pdb_ec_dict_and_ec_list(configs.dataset.annotation_path)
    elif configs.dataset.name == 'GO':
        pdb_ec_dict,_= get_pdb_go_dict_and_go_list(configs.dataset.annotation_path, configs.dataset.go_term)

    train_dataset = ProteinGraphDataset(configs.train_settings.data_path, 
                                        max_length = configs.model.esm_encoder.max_length,
                                        seq_mode=seq_mode,
                                        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                        rotary_mode=configs.model.struct_encoder.rotary_mode,
                                        use_foldseek=configs.model.struct_encoder.use_foldseek,
                                        use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
                                        top_k=configs.model.struct_encoder.top_k,
                                        num_rbf=configs.model.struct_encoder.num_rbf,
                                        num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings,
                                        apply_contrastive_learning=configs.model.contrastive_learning.apply,
                                        n_pos=configs.model.contrastive_learning.n_pos,
                                        n_neg=configs.model.contrastive_learning.n_neg,
                                        contrastive_sample_granularity=configs.model.contrastive_learning.contrastive_sample_granularity,
                                        dataset_name=configs.dataset.name,
                                        pdb_ec_dict=pdb_ec_dict,
                                        contact_cutoff=configs.model.struct_encoder.contact_cutoff
                                        )

    val_dataset = ProteinGraphDataset(configs.valid_settings.data_path, 
                                      max_length = configs.model.esm_encoder.max_length,
                                      seq_mode=seq_mode,
                                      use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                      rotary_mode=configs.model.struct_encoder.rotary_mode,
                                      use_foldseek=configs.model.struct_encoder.use_foldseek,
                                      use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
                                      top_k=configs.model.struct_encoder.top_k,
                                      num_rbf=configs.model.struct_encoder.num_rbf,
                                      num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings,
                                      apply_contrastive_learning=configs.model.contrastive_learning.apply,
                                      n_pos=configs.model.contrastive_learning.n_pos,
                                      n_neg=configs.model.contrastive_learning.n_neg,
                                      contrastive_sample_granularity=configs.model.contrastive_learning.contrastive_sample_granularity,
                                      dataset_name=configs.dataset.name,
                                      pdb_ec_dict=pdb_ec_dict,
                                      contact_cutoff=configs.model.struct_encoder.contact_cutoff
                                      )

    test_dataset = ProteinGraphDataset(configs.test_settings.data_path, 
                                      max_length = configs.model.esm_encoder.max_length,
                                      seq_mode=seq_mode,
                                      use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                      rotary_mode=configs.model.struct_encoder.rotary_mode,
                                      use_foldseek=configs.model.struct_encoder.use_foldseek,
                                      use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
                                      top_k=configs.model.struct_encoder.top_k,
                                      num_rbf=configs.model.struct_encoder.num_rbf,
                                      num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings,
                                      apply_contrastive_learning=False,
                                      dataset_name=configs.dataset.name,
                                      pdb_ec_dict=pdb_ec_dict,
                                      contact_cutoff=configs.model.struct_encoder.contact_cutoff
                                      )

    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                              shuffle=configs.train_settings.shuffle,
                              num_workers=configs.train_settings.num_workers,
                              multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=True,
                              collate_fn=custom_collate_contrastive_learning)  # Yichuan

    val_loader = DataLoader(val_dataset, batch_size=configs.valid_settings.batch_size,
                            num_workers=configs.valid_settings.num_workers,
                            shuffle=False,
                            multiprocessing_context='spawn' if configs.valid_settings.num_workers > 0 else None,
                            pin_memory=True,
                            collate_fn=custom_collate_contrastive_learning)

    test_loader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size,
                            num_workers=configs.valid_settings.num_workers,
                            shuffle=False,
                            multiprocessing_context='spawn' if configs.valid_settings.num_workers > 0 else None,
                            pin_memory=True,
                            collate_fn=custom_collate_contrastive_learning)


    return train_loader, val_loader,test_loader


if __name__ == '__main__':

    dataset_path = '/data/duolin/CATH/CATH-pos-512-gvp_splitdata/val/' #'../../swiss-pos-512-gvp'
    dataset = ProteinGraphDataset(dataset_path)

    test_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=custom_collate)

    print("Testing on your dataset")
    for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader)):
        graph = batch["graph"].to('cuda')  # Move the batch to the device
        h_V = (graph.node_s, graph.node_v)
        h_E = (graph.edge_s, graph.edge_v)
        e_E = graph.edge_index
        print(batch['seq'])
        print(batch['plddt'])
        # sample = model.sample(h_V, batch.edge_index, h_E, n_samples=100)
        # Continue with the rest of your processing...

    print('done')
