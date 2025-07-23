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

class ProteinGraphDataset(Dataset):
    """
    This class is a subclass of `torch.utils.data.Dataset` and is used to transform JSON/dictionary-style
    protein structures into featurized protein graphs. The transformation process is described in detail in the
    associated manuscript.

    The transformed protein graphs are instances of `torch_geometric.data.Data` and have the following attributes:
    - x: Alpha carbon coordinates. This is a tensor of shape [n_nodes, 3].
    - seq: Protein sequence converted to an integer tensor according to `self.letter_to_num`. This is a tensor of shape [n_nodes].
    - name: Name of the protein structure. This is a string.
    - node_s: Node scalar features. This is a tensor of shape [n_nodes, 6].
    - node_v: Node vector features. This is a tensor of shape [n_nodes, 3, 3].
    - edge_s: Edge scalar features. This is a tensor of shape [n_edges, 32].
    - edge_v: Edge scalar features. This is a tensor of shape [n_edges, 1, 3].
    - edge_index: Edge indices. This is a tensor of shape [2, n_edges].
    - mask: Node mask. This is a boolean tensor where `False` indicates nodes with missing data that are excluded from message passing.

    This class uses portions of code from https://github.com/jingraham/neurips19-graph-protein-design.

    Parameters:
    - data_list: directory of h5 files.
    - num_positional_embeddings: The number of positional embeddings to use.
    - top_k: The number of edges to draw per node (as destination node).
    - device: The device to use for preprocessing. If "cuda", preprocessing will be done on the GPU.
    """

    def __init__(self, data_path,max_length=512,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cuda",
                 seq_mode="embedding", use_rotary_embeddings=False,rotary_mode=1,
                 use_foldseek = False,use_foldseek_vector=False
                 ):
        super(ProteinGraphDataset, self).__init__()

        self.h5_samples = data_path
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.seq_mode = seq_mode
        # self.node_counts = [len(e['seq']) for e in data_list]
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rotary_mode = rotary_mode
        self.use_foldseek = use_foldseek
        self.use_foldseek_vector = use_foldseek_vector
        self.max_length = max_length
        if self.use_rotary_embeddings:
           if self.rotary_mode==3:
              self.rot_emb = RotaryEmbedding(dim=8)#must be 5
           else:
              self.rot_emb = RotaryEmbedding(dim=2)#must be 2
        
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

    def __len__(self):
        return len(self.h5_samples)

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sequence,coord_list = load_h5_file(sample_path)
        raw_seqs = sequence.decode('utf-8')
        basename = os.path.basename(sample_path)
        pid = "_".join(basename.split('_')[:2])
        # pid = basename.split('.h5')[0]
        feature_list=[]
        for coord in coord_list:
            coord = coord[:self.max_length]
            sample_dict = {'name': pid,
                           'coords': coord.tolist(),
                           'seq': sequence.decode('utf-8')}
            feature = self._featurize_as_graph(sample_dict)
            feature_list.appen(feature)
        # plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
        
        #coordinates = torch.as_tensor(sample[1],dtype=torch.float32)
        return [feature_list, raw_seqs, pid]

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
            elif self.seq_mode == "noseq":
                seq = None
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            if self.use_rotary_embeddings:
                if self.rotary_mode==1: #first mode
                   d1 = edge_index[0] - edge_index[1]
                   d2 = edge_index[1] - edge_index[0]
                   d = torch.cat((d1.unsqueeze(-1),d2.unsqueeze(-1)),dim=-1) #[len,2]
                   pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
                if self.rotary_mode==2:
                    d = edge_index.transpose(0,1)
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
                if self.rotary_mode==3:
                    d = edge_index.transpose(0,1) #[len,2]
                    d = torch.cat((d,E_vectors,-1*E_vectors),dim=-1) #[len,2+3]
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
            else:
                pos_embeddings = self._positional_embeddings(edge_index)
            
            rbf = _rbf(E_vectors.norm(dim=-1), d_count=self.num_rbf,
                       # device=self.device
                       )

            dihedrals = self._dihedrals(coords)  # only this one used O
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            if self.use_foldseek or self.use_foldseek_vector:
                foldseek_features = self._foldseek(X_ca)
            
            if self.use_foldseek:
                node_s = torch.cat([dihedrals,foldseek_features[0]],dim=-1) #add additional 10 features
            else:
                node_s = dihedrals
            
            if self.use_foldseek_vector:
               node_v = torch.cat([orientations, sidechains.unsqueeze(-2),foldseek_features[1]], dim=-2) #add additional 18 features
            else:
               node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

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
       #From Fast and accurate protein structure search with Foldseek
       #x is X_ca coordinates
       x=torch.nan_to_num(x,nan=0.0,posinf=0.0, neginf=0.0)
       j,i = torch_cluster.knn_graph(x, k=1)
       mask = torch.zeros_like(i, dtype=torch.bool)
       first_unique = {}
       for index,value in enumerate(i):
            if value.item() not in first_unique:
               first_unique[value.item()] = index
               mask[index]=True
            else:
               mask[index]=False
       
       i,j = i[mask],j[mask]
       pad_x = F.pad(x, [0,0,1,1]) #padding to the first and last
       j = j+1
       i = i+1
       d = torch.norm(pad_x[i]-pad_x[j],dim=-1,keepdim=True)
       abs_diff = torch.abs(i-j)
       sign = torch.sign(i-j)
       f1 = (sign * torch.minimum(abs_diff,torch.tensor(4))).unsqueeze(-1)
       f2 = (sign * torch.log(abs_diff+1)).unsqueeze(-1)
       
       u1 = _normalize(pad_x[i]-pad_x[i-1])
       u2 = _normalize(pad_x[i+1]-pad_x[i])
       u3 = _normalize(pad_x[j]-pad_x[j-1])
       u4 = _normalize(pad_x[j+1]-pad_x[j])
       u5 = _normalize(pad_x[j]-pad_x[i])
       cos_12 = torch.sum(u1 * u2, dim=-1,keepdim=True)
       cos_34 = torch.sum(u3 * u4, dim=-1,keepdim=True)
       cos_15 = torch.sum(u1 * u5, dim=-1,keepdim=True)
       cos_35 = torch.sum(u3 * u5, dim=-1,keepdim=True)
       cos_14 = torch.sum(u1 * u4, dim=-1,keepdim=True)
       cos_23 = torch.sum(u2 * u3, dim=-1,keepdim=True)
       cos_13 = torch.sum(u1 * u3, dim=-1,keepdim=True)
       node_s_features = torch.cat([cos_12,cos_34,cos_15,cos_35,cos_14,cos_23,cos_13,d,f1,f2],dim=-1)
       node_v_features = torch.cat([u1.unsqueeze(-2),u2.unsqueeze(-2),u3.unsqueeze(-2),
                                    u4.unsqueeze(-2),u5.unsqueeze(-2),(pad_x[i]-pad_x[j]).unsqueeze(-2)],dim=-2) #add 6 additional features
       return node_s_features,node_v_features

def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        # plddt_scores = f['plddt_scores'][:]
    return seq, n_ca_c_o_coord

class ProteinMDDataset(Dataset):
    def __init__(self, samples, configs,mode="train"):
        # self.original_samples = samples
        self.file_pairs = samples  # list of (traj_path, h5_path)
        self.configs = configs
        self.maxlen = configs.model.esm_encoder.max_length
    def __len__(self):
           return len(self.file_pairs)
    def __getitem__(self, idx):
        # pid, seq, res_rep = self.original_samples[idx]
        traj_path, h5_path = self.file_pairs[idx]
        traj_rep = torch.load(traj_path) # [T, M, 4, D]
        traj_rep = traj_rep.permute(1, 0, 2, 3) #[M, T, 4, D]
        traj_rep = traj_rep.reshape(traj_rep.shape[0], traj_rep.shape[1], -1)  # → [M, T, 4*D]
        # bin_res_rep = bin_average_functional(traj_rep, n_bins=100) #[M, 100, dim]
        bin_res_rep = chunk_bin(traj_rep, chunk_size=100, bin_number=100) #[M, 100, dim]
        geom2vec, seq, pid = load_h5(h5_path)
        return pid, seq, bin_res_rep[:self.maxlen] #res_rep [n_residue, T, 512]

import torch.nn.functional as F


def custom_collate(one_batch):
    # Unpack the batch
    torch_geometric_list_list = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data
    torch_geometric_list=[]
    for t in range(100):
        torch_geometric_batch = Batch.from_data_list([gl[t] for gl in torch_geometric_list_list])
        torch_geometric_list.append(torch_geometric_batch)
    
    # Create a Batch object
    # torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    raw_seqs = [item[1] for item in one_batch]
    # plddt_scores = [item[2] for item in one_batch]
    pids = [item[2] for item in one_batch]
    # plddt_scores = torch.cat(plddt_scores, dim=0)
    batched_data = {'graph': torch_geometric_list, 'seq': raw_seqs, 'plddt': plddt_scores,'pid': pids}
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

    # v2add_samples = prepare_samples(Atlas_v2add_path, configs)
    # train_samples = train_samples + v2add_samples
    
    print(f"train samples: {len(train_samples)}, val samples: {len(val_samples)}, test samples: {len(test_samples)}")
    # Create DataLoader for each split
    # train_dataset = ProteinMDDataset(train_samples, configs=configs, mode="train")
    train_dataset = ProteinGraphDataset(train_samples,
                                        max_length = configs.model.esm_encoder.max_length,seq_mode=seq_mode,
                                        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                        rotary_mode = configs.model.struct_encoder.rotary_mode,
                                        use_foldseek = configs.model.struct_encoder.use_foldseek,
                                        use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                        top_k = configs.model.struct_encoder.top_k,
                                        num_rbf = configs.model.struct_encoder.num_rbf,
                                        num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings,
                                        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.train_settings.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=False
    )
    # val_dataset = ProteinMDDataset(val_samples, configs=configs, mode="val")
    val_dataset = ProteinGraphDataset(val_samples, 
                                      max_length = configs.model.esm_encoder.max_length,seq_mode=seq_mode,
                                      use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                      rotary_mode = configs.model.struct_encoder.rotary_mode,
                                      use_foldseek = configs.model.struct_encoder.use_foldseek,
                                      use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                      top_k = configs.model.struct_encoder.top_k,
                                      num_rbf = configs.model.struct_encoder.num_rbf,
                                      num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings,
                                      )
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
    # train_dataloader, val_dataloader, test_dataloader = prepare_replicate(configs, 
    #                                                                       configs.train_settings.Atlas_data_path, 
    #                                                                       configs.train_settings.Atlas_test_path,
    #                                                                       configs.train_settings.Atlas_v2add_path)
  
    # return train_dataloader, val_dataloader, test_dataloader
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
        if file_name.endswith(".h5"):
            file_path = os.path.abspath(os.path.join(datapath, file_name))
            # traj_rep = torch.load(file_path) # [T, M, 4, D]
            # traj_rep = traj_rep.permute(1, 0, 2, 3) #[M, T, 4, D]
            # traj_rep = traj_rep.reshape(traj_rep.shape[0], traj_rep.shape[1], -1)  # → [M, T, 4*D]
            # # bin_res_rep = bin_average_functional(traj_rep, n_bins=100) #[M, 100, dim]
            # bin_res_rep = chunk_bin(traj_rep, chunk_size=100, bin_number=100) #[M, 100, dim]
            
            # without_extension=os.path.splitext(file_name)[0]
            # h5_filename = os.path.join(datapath, f"{without_extension}.h5")
            # geom2vec, seq, pid = load_h5(h5_filename)
        else:
            continue
        # prot_rep = geom2vec.mean
        # samples.append((pid, seq, bin_res_rep[:maxlen])) #bin_res_rep [n_residue, 1000, dim]
        samples.append(file_path)
    
    return samples


def load_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        geom2vec = f['geom2vec'][:]
        pid = f['pid'][()].decode('utf-8')
        seq = f['seq'][()].decode('utf-8')  # Decode from bytes to string
    
    return geom2vec, seq, pid

  



