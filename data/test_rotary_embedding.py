import torch
d = torch.randn(10, 1)

def _update_cos_sin_tables(x, seq_dimension=1,inv_freq=None):
        seq_len = x.shape[seq_dimension]
        
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        _seq_len_cached = seq_len
        
        t = torch.arange(x.shape[seq_dimension]).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t,inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        _cos_cached = emb.cos()[None, :, :]
        _sin_cached = emb.sin()[None, :, :]
        
        return _cos_cached, _sin_cached



def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :] #1,10,16
    sin = sin[:, : x.shape[-2], :]
    
    return (x * cos) + (rotate_half(x) * sin)

def rotate_half(x):  #10,2
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

d = torch.randn(1,10,1,2)
#d = d.repeat(1, 1, 1, 16)

#d = torch.randn(32,256,12,64)
#dim=64
d = torch.randn(1,10,1,1)
d = torch.torch.cat((d,d),dim=-1) #[len,2]


import torch_cluster
X_ca = torch.randn(10,3)
edge_index = torch_cluster.knn_graph(X_ca, k=5)
E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
d = edge_index.transpose(0,1) #[len,2]
d = torch.cat((d,E_vectors),dim=-1) #[len,2+3]
d=d.unsqueeze(0).unsqueeze(-2)
dim=d.shape[-1] #must be 2 as same as d

inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
_cos_cached, _sin_cached = _update_cos_sin_tables(d, seq_dimension=-2,inv_freq=inv_freq)
#1,10,16 1,10,16
#1,12,64
d_rot=apply_rotary_pos_emb(d, _cos_cached, _sin_cached)

d_rot.shape = 32,256,12,64!!!

batch_size, sequence_length, heads, head_dimension