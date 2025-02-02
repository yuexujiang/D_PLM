import glob
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import tqdm
import os
from torch.utils.data import DataLoader, Dataset
from .utils import load_h5_file_v2
from torch_geometric.data import Batch, Data
import time
from .rotary_embedding import RotaryEmbedding
epsilon = 1e-10
def merge_features_and_create_mask(features_list, max_length=512):
    # Pad tensors and create mask
    padded_tensors = []
    mask_tensors = []
    for t in features_list:
        if t.size(0) < max_length:
            size_diff = max_length - t.size(0)
            pad = torch.zeros(size_diff, t.size(1), device=t.device)
            t_padded = torch.cat([t, pad], dim=0)
            mask = torch.cat([torch.ones(t.size(0), dtype=torch.bool, device=t.device),
                              torch.zeros(size_diff, dtype=torch.bool, device=t.device)], dim=0)
        else:
            t_padded = t
            mask = torch.ones(t.size(0), dtype=torch.bool, device=t.device)
        padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation
        mask_tensors.append(mask.unsqueeze(0))  # Add an extra dimension for concatenation
    
    # Concatenate tensors and masks
    result = torch.cat(padded_tensors, dim=0)
    mask = torch.cat(mask_tensors, dim=0)
    return result, mask


def custom_collate(one_batch):
    # Unpack the batch
    torch_geometric_feature = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data

    # Create a Batch object
    torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    raw_seqs = [item[1] for item in one_batch]
    plddt_scores = [item[2] for item in one_batch]
    pids = [item[3] for item in one_batch]
    plddt_scores = torch.cat(plddt_scores, dim=0)
    batched_data = {'graph': torch_geometric_batch, 'seq': raw_seqs, 'plddt': plddt_scores,'pid': pids}
    return batched_data


def custom_collate_mask(one_batch):
    # Unpack the batch
    torch_geometric_feature = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data
    torch_geometric_featuremask = [item[1] for item in one_batch]  # item[0] is for torch_geometric Data
    coordinates  = torch.stack([item[2] for item in one_batch])
    coordinates_mask  = torch.stack([item[3] for item in one_batch])
    # Create a Batch object
    torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    torch_geometric_batch_mask = Batch.from_data_list(torch_geometric_featuremask)
    raw_seqs = [item[4] for item in one_batch]
    plddt_scores = [item[5] for item in one_batch]
    pids = [item[6] for item in one_batch]
    plddt_scores = torch.cat(plddt_scores, dim=0)
    masks = torch.stack([item[7] for item in one_batch])
    #print("coordinates shape in custom")
    #print(coordinates.shape)
    batched_data = {'graph': torch_geometric_batch, 'mask_graph':torch_geometric_batch_mask,'seq': raw_seqs, 'plddt': plddt_scores, 'pid': pids,'coordinates':coordinates,'coordinates_mask':coordinates_mask,"masks":masks}
    
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

def handle_nan_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """
    Replaces NaN values in the coordinates with the previous or next valid coordinate values.

    Parameters:
    -----------
    coords : torch.Tensor
        A tensor of shape (N, 4, 3) representing the coordinates of a protein structure.

    Returns:
    --------
    torch.Tensor
        The coordinates with NaN values replaced by the previous valid coordinate values.
    """
    # Flatten the coordinates for easier manipulation
    original_shape = coords.shape
    coords = coords.view(-1, 3)

    # check if there are any NaN values in the coordinates
    while torch.isnan(coords).any():
        # Identify NaN values
        nan_mask = torch.isnan(coords)

        if not nan_mask.any():
            return coords.view(original_shape)  # Return if there are no NaN values

        # Iterate through coordinates and replace NaNs with the previous valid coordinate
        for i in range(1, coords.shape[0]):
            if nan_mask[i].any() and not torch.isnan(coords[i - 1]).any():
                coords[i] = coords[i - 1]

        for i in range(0, coords.shape[0] - 1):
            if nan_mask[i].any() and not torch.isnan(coords[i + 1]).any():
                coords[i] = coords[i + 1]

    return coords.view(original_shape)


def eight2three(ss):
    if ss=='H':
        return [1,0,0,0,0,0,0,0]
    elif ss=='G':
        return [0,1,0,0,0,0,0,0]
    elif ss=='I':
        return [0,0,1,0,0,0,0,0]
    elif ss=='E':
        return [0,0,0,1,0,0,0,0]
    elif ss=='B':
        return [0,0,0,0,1,0,0,0]
    elif ss=='T':
        return [0,0,0,0,0,1,0,0]
    elif ss=='S':
        return [0,0,0,0,0,0,1,0]
    else:
        return [0,0,0,0,0,0,0,1]

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
    
    def __init__(self,data_path,max_length=512,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cuda",
                 seq_mode="embedding", use_rotary_embeddings=False,rotary_mode=1,
                 use_foldseek = False,use_foldseek_vector=False
                 ):
        
        super(ProteinGraphDataset, self).__init__()

        self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))
        sequence,coord,plddt_scores,ss,sasa,node_degree,edge_index,edge_dis,edge_localorientation= load_h5_file_v2(self.h5_samples[0])
        
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
        sequence,coord,plddt_scores,ss,sasa,node_degree,edge_index,edge_dis,edge_localorientation= load_h5_file_v2(sample_path)
        coord = coord[:self.max_length]
        basename = os.path.basename(sample_path)
        pid = basename.split('.h5')[0]

        sample_dict = {'name': pid,
                       'coords': coord, #.tolist(), no need to convert into list can handle numpy in torch.as_tensor
                       'seq': sequence.decode('utf-8'),
                       'ss':  ss,
                       'sasa':sasa,
                       'node_degree':node_degree,
                       'edge_index':edge_index,
                       'edge_dis':edge_dis,
                       'edge_localorientation':edge_localorientation,
                       }
        feature = self._featurize_as_graph(sample_dict)
        plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
        raw_seqs = sequence.decode('utf-8')
        #coordinates = torch.as_tensor(sample[1],dtype=torch.float32)
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
            elif self.seq_mode == "noseq":
                seq = None
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            #edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            edge_index = torch.as_tensor(protein['edge_index'],dtype=torch.long)
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

            dihedrals = self._dihedrals(coords)  # only this one used O changed into 12D by adding sin(2D) cos(2D)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            ss_onehot = torch.as_tensor([eight2three(x) for x in protein['ss']],dtype=torch.float32)
            sasa = torch.as_tensor(protein['sasa'],dtype = torch.float32)
            node_degree = torch.as_tensor(protein['node_degree'],dtype = torch.float32)
            edge_dis = torch.as_tensor(protein['edge_dis'],dtype = torch.float32)
            edge_localorientation = torch.as_tensor(protein['edge_localorientation'],dtype = torch.float32)
            if self.use_foldseek or self.use_foldseek_vector:
                foldseek_features = self._foldseek(X_ca)
            
            if self.use_foldseek: #scalar node features, added 11 more! dihedrals add additional 6D
                node_s = torch.cat([dihedrals,ss_onehot,sasa.unsqueeze(-1),node_degree,foldseek_features[0]],dim=-1) #add additional 10 features
            else:
                node_s = torch.cat([dihedrals,ss_onehot,sasa.unsqueeze(-1),node_degree],dim=-1) #add 6+8+1+2 = 17 more features
            
            if self.use_foldseek_vector:
               node_v = torch.cat([orientations, sidechains.unsqueeze(-2),foldseek_features[1]], dim=-2) #add additional 18 features
            else:
               node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            
            edge_s = torch.cat([rbf, pos_embeddings,edge_dis,1/((edge_dis+epsilon) ** 2),1/((edge_dis+epsilon) ** 4),1/((edge_dis+epsilon) ** 6),edge_localorientation], dim=-1) #added 7 more
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
        #D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        D_features = torch.cat([torch.cos(D),torch.cos(2*D),torch.sin(D),torch.sin(2*D)], 1)
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


class ProteinGraphDataset_mask(ProteinGraphDataset):
    def __init__(self, data_path,max_length,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cuda",
                 seq_mode="embedding", use_rotary_embeddings=False,rotary_mode=1,
                 use_foldseek = False,use_foldseek_vector=False,mlm_probability=0.25):
        
        super(ProteinGraphDataset_mask, self).__init__(data_path,max_length,num_positional_embeddings,
                 top_k,num_rbf, device,
                 seq_mode,use_rotary_embeddings,rotary_mode,
                 use_foldseek,use_foldseek_vector)
        self.mlm_probability = mlm_probability
        self.max_length = max_length
    """
    def mask_coords(self,coords):
        inputs = coords.clone().to(coords.device) #coords L,3,4
        #labels = coords.clone().to(coords.device)
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        indices_replaced = torch.bernoulli(probability_matrix).bool()
        inputs[indices_replaced] = 0.0
        return inputs #, labels
    """
    #to do mask residue index all the other features precalculated as sasa,sa need to be masked!
    def mask_coords(self, coords):
        inputs = coords.clone().to(coords.device)  # coords of shape [L, 3, 4]
        # Create a mask for the first part of the coordinates [L, 3, 0]
        probability_matrix = torch.full((coords.shape[0], coords.shape[1], 1), self.mlm_probability, device=coords.device)
        indices_replaced = torch.bernoulli(probability_matrix).bool()
        # Replicate the mask to match the shape of [L, 3, 4]
        expanded_mask = indices_replaced.expand_as(coords)
        # Apply the mask to all parts of the coordinates [L, 3, 0] and [L, 3, 1:4]
        inputs[expanded_mask] = 0.0
        
        return inputs
    def mask_tensor(self, tensor):
        # Clone the tensor to avoid modifying the original data
        masked_tensor = tensor.clone().to(tensor.device)
        # Create a probability matrix and generate mask based on mlm_probability
        probability_matrix = torch.full(tensor.shape, self.mlm_probability, device=tensor.device)
        mask = torch.bernoulli(probability_matrix).bool()
        # Set masked elements to zero
        masked_tensor[mask] = 0.0
        return masked_tensor
    def mask_gvp_feature(self, data):
        # Apply masking to each relevant tensor in the Data object
        masked_node_s = self.mask_tensor(data.node_s) if 'node_s' in data else None
        masked_node_v = self.mask_tensor(data.node_v) if 'node_v' in data else None
        masked_edge_s = self.mask_tensor(data.edge_s) if 'edge_s' in data else None
        masked_edge_v = self.mask_tensor(data.edge_v) if 'edge_v' in data else None
        # Create a new Data object with masked values, preserving other attributes
        data_masked = Data(
            x=data.x,
            seq=data.seq,
            name=data.name,
            node_s=masked_node_s,
            node_v=masked_node_v,
            edge_s=masked_edge_s,
            edge_v=masked_edge_v,
            edge_index=data.edge_index,
            mask=data.mask
        )
        return data_masked

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sequence,coord,plddt_scores  = load_h5_file_v2(sample_path)
        coord = coord[:self.max_length]
        basename = os.path.basename(sample_path)
        pid = basename.split('.h5')[0]
        sample_dict = {'name': pid,
                       'coords': coord.tolist(),
                       'seq': sequence.decode('utf-8'),
                       'ss':  ss,
                       'sasa':sasa,
                       'node_degree':node_degree,
                       'edge_index':edge_index,
                       'edge_dis':edge_dis,
                       'edge_localorientation':edge_localorientation,
                       }
        feature = self._featurize_as_graph(sample_dict)
        plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
        raw_seqs = sequence.decode('utf-8')
        coordinates = torch.as_tensor(coord,dtype=torch.float32)
        coordinates=handle_nan_coordinates(coordinates)
        coordinates_mask = coordinates.clone().to(coordinates.device) #self.mask_coords(coordinates)
        """ for data_v2 cannot do this anymore, can mask directly on gvp features from Data object
        mask_dict = {'name': pid,
             'coords': coordinates_mask.tolist(),
             'seq': sequence.decode('utf-8')}
        mask_feature = self._featurize_as_graph(mask_dict)
        """
        mask_feature = mask_gvp_feature(feature)
        coordinates,masks = merge_features_and_create_mask(coordinates.reshape(1, -1, 12),self.max_length)
        masks = masks.squeeze(0)
        coordinates_mask,_ = merge_features_and_create_mask(coordinates_mask.reshape(1, -1, 12),self.max_length)
        coordinates_mask=coordinates_mask.squeeze(0)
        coordinates=coordinates.squeeze(0)
        #print("coordinatesshape=")
        #print(coordinates.shape)
        return [feature, mask_feature, coordinates,coordinates_mask,raw_seqs, plddt_scores, pid,masks]


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
                                        rotary_mode = configs.model.struct_encoder.rotary_mode,
                                        use_foldseek = configs.model.struct_encoder.use_foldseek,
                                        use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                        top_k = configs.model.struct_encoder.top_k,
                                        num_rbf = configs.model.struct_encoder.num_rbf,
                                        num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings,
                                        )
    val_dataset = ProteinGraphDataset(configs.valid_settings.data_path, 
                                      max_length = configs.model.esm_encoder.max_length,seq_mode=seq_mode,
                                      use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                      rotary_mode = configs.model.struct_encoder.rotary_mode,
                                      use_foldseek = configs.model.struct_encoder.use_foldseek,
                                      use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                      top_k = configs.model.struct_encoder.top_k,
                                      num_rbf = configs.model.struct_encoder.num_rbf,
                                      num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings,
                                      )
    
    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                              shuffle=configs.train_settings.shuffle,
                              num_workers=configs.train_settings.num_workers,
                              multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=True,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=configs.valid_settings.batch_size,
                            num_workers=configs.valid_settings.num_workers,
                            shuffle=False,
                            multiprocessing_context='spawn' if configs.valid_settings.num_workers > 0 else None,
                            pin_memory=True,
                            collate_fn=custom_collate)
    return train_loader, val_loader

def prepare_mask_dataloaders(logging, accelerator, configs):
    if accelerator.is_main_process:
        logging.info(f"train directory: {configs.train_settings.data_path}")
        logging.info(f"valid directory: {configs.valid_settings.data_path}")
    
    if hasattr(configs.model.struct_encoder, "use_seq") and configs.model.struct_encoder.use_seq.enable:
        seq_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
    else:
        seq_mode = "noseq"
    
    train_dataset = ProteinGraphDataset_mask(configs.train_settings.data_path, 
                                        max_length = configs.model.esm_encoder.max_length,
                                        seq_mode=seq_mode,
                                        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                        rotary_mode = configs.model.struct_encoder.rotary_mode,
                                        use_foldseek = configs.model.struct_encoder.use_foldseek,
                                        use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                        top_k = configs.model.struct_encoder.top_k,
                                        num_rbf = configs.model.struct_encoder.num_rbf,
                                        num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings,
                                        mlm_probability = configs.model.struct_decoder.mlm_prob,
                                        )
    val_dataset = ProteinGraphDataset_mask(configs.valid_settings.data_path, 
                                      max_length = configs.model.esm_encoder.max_length,
                                      seq_mode=seq_mode,
                                      use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                      rotary_mode = configs.model.struct_encoder.rotary_mode,
                                      use_foldseek = configs.model.struct_encoder.use_foldseek,
                                      use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                      top_k = configs.model.struct_encoder.top_k,
                                      num_rbf = configs.model.struct_encoder.num_rbf,
                                      num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings,
                                      mlm_probability = configs.model.struct_decoder.mlm_prob,
                                      )
    
    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                              shuffle=configs.train_settings.shuffle,
                              num_workers=configs.train_settings.num_workers,
                              multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=True,
                              collate_fn=custom_collate_mask)
    val_loader = DataLoader(val_dataset, batch_size=configs.valid_settings.batch_size,
                            num_workers=configs.valid_settings.num_workers,
                            shuffle=False,
                            multiprocessing_context='spawn' if configs.valid_settings.num_workers > 0 else None,
                            pin_memory=True,
                            collate_fn=custom_collate_mask)
    return train_loader, val_loader



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
