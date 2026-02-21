import torch.nn as nn
import torch
import torch.nn.functional as F
import esm
import esm_adapterH
from peft import PeftModel, LoraConfig, get_peft_model
import gvp.models
# from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
import numpy as np
from torch_scatter import scatter_mean
from transformers import VivitImageProcessor, VivitModel
from transformers.models.vivit.modeling_vivit import VivitLayer
from typing import Optional, Tuple
from transformers import AutoModel

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for batch and sequence length
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pe[:x.size(0), :]
        return x


def patch_num_32(n):
    remainder = n % 32
    if remainder == 0:
        return n // 32
    else:
        return (n + (32 - remainder)) // 32


def save_lora_checkpoint(model, save_path):
    """
    Save lora weights as a checkpoint in the save_path directory.
    """
    model.save_pretrained(save_path)


def load_and_add_lora_checkpoint(base_model, lora_checkpoint_path):
    """Add a pretrained LoRa checkpoint to a base model"""
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    return lora_model


def merge_lora_with_base_model(lora_model):
    """
    This method merges the LoRa layers into the base model.
    This is needed if someone wants to use the base model as a standalone model.
    """
    model = lora_model.merge_and_unload()
    return model


def remove_lora(lora_model):
    """
    Gets back the base model by removing all the lora modules without merging
    """
    base_model = lora_model.unload()
    return base_model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_negative_mean_logtis(logits, mode, minibatch_size):
    N = minibatch_size
    if mode == "struct_struct":
        a11 = logits[0:N, 1:N]
        return a11.mean().item()

    if mode == "struct_seq":
        a12 = logits[0:N, N:2 * N - 1]
        return a12.mean().item()

    if mode == "seq_struct":
        a21 = logits[N:2 * N, 1:N]
        return a21.mean().item()
    if mode == "seq_seq":
        a22 = logits[N:2 * N, N:2 * N - 1]
        return a22.mean().item()


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        #print("mlp forward")
        #print(x.shape)  #[128,512,100]
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x


class GVPEncoder(nn.Module):  # embedding table can be tuned
    def __init__(self, configs=None, residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2,
                 seqlen=512):
        super(GVPEncoder, self).__init__()
        if hasattr(configs.model.struct_encoder,"version") and configs.model.struct_encoder.version == 'v2':
              node_in_dim = [12+11, 3] ##use data_v2.py to extract gvp features
        else:
            node_in_dim = [6, 3] #default
        
        if configs.model.struct_encoder.use_foldseek:
            node_in_dim[0] += 10 #foldseek has 10 more node scalar features
        
        if configs.model.struct_encoder.use_foldseek_vector:
            node_in_dim[1] += 6 #foldseek_vector has 6 more node vector features
        
        node_in_dim = tuple(node_in_dim)
        if configs.model.struct_encoder.use_rotary_embeddings:
            if configs.model.struct_encoder.rotary_mode==3:
                edge_in_dim = (configs.model.struct_encoder.num_rbf+8,1) #16+2+3+3 only for mode ==3 add 8D pos_embeddings
            else: 
                edge_in_dim = (configs.model.struct_encoder.num_rbf+2,1) #16+2 
        else:
              edge_in_dim = (configs.model.struct_encoder.num_rbf+configs.model.struct_encoder.num_positional_embeddings, 1) #num_rbf+num_positional_embeddings
        
        if hasattr(configs.model.struct_encoder,"version") and configs.model.struct_encoder.version == 'v2':
            edge_in_dim = (edge_in_dim[0]+7,1) #in v2 add 7 more D
        
        node_h_dim = configs.model.struct_encoder.node_h_dim
        # node_h_dim=(100, 16) #default
        #node_h_dim = (100, 32)  # seems best?
        edge_h_dim=configs.model.struct_encoder.edge_h_dim
        #edge_h_dim = (32, 1) #default
        gvp_num_layers = configs.model.struct_encoder.gvp_num_layers
        #gvp_num_layers = 3
        
        self.use_seq = configs.model.struct_encoder.use_seq.enable
        if self.use_seq:
            self.seq_embed_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
            self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=True,
                                                         seq_embed_mode=self.seq_embed_mode,
                                                         seq_embed_dim=configs.model.struct_encoder.use_seq.seq_embed_dim,
                                                         num_layers=gvp_num_layers)
        else:
            self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=False,
                                                         num_layers=gvp_num_layers)

        # pretrained_state_dict = init_model.state_dict()
        # self.backbone.load_state_dict(pretrained_state_dict,strict=False)

        dim_mlp = node_h_dim[0]
        # self.esm_pool_mode= configs.model.esm_encoder.pool_mode
        self.projectors_protein = MoBYMLP(in_dim=dim_mlp, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        self.projectors_residue = MoBYMLP(in_dim=dim_mlp, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
                                          num_layers=residue_num_projector)
        

        if hasattr(configs.model.struct_encoder, "fine_tuning") and not configs.model.struct_encoder.fine_tuning.enable:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        if hasattr(configs.model.struct_encoder, "fine_tuning_projct") and not configs.model.struct_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False
            
            for name, param in self.projectors_residue.named_parameters():
                param.requires_grad = False
        
        # for p in self.backbone.parameters():
        #    p.requires_grad = False  # fix all previous layers

        # for name, module in model.named_modules():
        #     print(f"Module Name: {name}")
        #     print(module)
        #     print("=" * 50)

    def forward(self, graph, esm2_representation=None,
                return_embedding=False):  # this batch is torch_geometric batch.batch to indicate the batch
        """
        graph: torch_geometric batchdasta
        """
        #print("graph.node shapes")
        #print(graph.node_s.shape) #torch.Size([20650, 6])         torch.Size([16865, 23])
        #print(graph.node_v.shape) #torch.Size([20650, 9, 3])      torch.Size([16865, 9, 3])
        #print(graph.edge_s.shape) #torch.Size([619500, 18])     torch.Size([443846, 25])
        #print(graph.edge_v.shape) #torch.Size([619500, 1, 3])    torch.Size([443846, 1, 3])
        
        nodes = (graph.node_s, graph.node_v)
        edges = (graph.edge_s, graph.edge_v)
        if self.use_seq and self.seq_embed_mode != "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=graph.seq)
        elif self.use_seq and self.seq_embed_mode == "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=esm2_representation #`torch.Tensor` of shape [num_nodes]
                                                      )
        else:
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=None)

        # graph_feature=scatter_mean(residue_feature, batch, dim=0)
        graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
        graph_feature = self.projectors_protein(graph_feature_embedding)
        residue_feature = self.projectors_residue(residue_feature_embedding)
        if return_embedding:
            return graph_feature, residue_feature, graph_feature_embedding, residue_feature_embedding
        else:
            return graph_feature, residue_feature

class GVP_lstm(nn.Module):  
    def __init__(self, 
                 configs,
                 input_dim=512, 
                 hidden_dim=128, 
                 output_dim=32,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, 
                 protein_num_projector=2,
                 lstm_layers=2,
                 dropout_rate=0.1):
        """
        BiLSTM-based temporal encoder with attention mechanism
        
        Args:
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate for LSTM and attention
        """
        super(GVP_lstm, self).__init__()
        node_in_dim = [6, 3] #default
        if configs.model.struct_encoder.use_foldseek:
            node_in_dim[0] += 10 #foldseek has 10 more node scalar features
        if configs.model.struct_encoder.use_foldseek_vector:
            node_in_dim[1] += 6 #foldseek_vector has 6 more node vector features
        node_in_dim = tuple(node_in_dim)
        if configs.model.struct_encoder.use_rotary_embeddings:
            if configs.model.struct_encoder.rotary_mode==3:
                edge_in_dim = (configs.model.struct_encoder.num_rbf+8,1) #16+2+3+3 only for mode ==3 add 8D pos_embeddings
            else: 
                edge_in_dim = (configs.model.struct_encoder.num_rbf+2,1) #16+2 
        else:
              edge_in_dim = (configs.model.struct_encoder.num_rbf+configs.model.struct_encoder.num_positional_embeddings, 1) #num_rbf+num_positional_embeddings
        node_h_dim = configs.model.struct_encoder.node_h_dim
        # node_h_dim=(100, 16) #default
        #node_h_dim = (100, 32)  # seems best?
        edge_h_dim=configs.model.struct_encoder.edge_h_dim
        #edge_h_dim = (32, 1) #default
        gvp_num_layers = configs.model.struct_encoder.gvp_num_layers
        #gvp_num_layers = 3
        self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=False,
                                                         num_layers=gvp_num_layers)
        
        dim_mlp = node_h_dim[0]



        # Input projection to reduce dimensionality before LSTM
        self.input_projection = nn.Linear(dim_mlp, hidden_dim)
        
        # Bidirectional LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=output_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # LSTM output dimension (bidirectional doubles the size)
        lstm_output_dim = output_dim * 2
        
        # Attention mechanism
        self.attention = TemporalAttention(lstm_output_dim)
        
        # Batch normalization and dropout
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final projection dimension after attention
        proj_dim = lstm_output_dim
        
        # Projectors for protein and residue level representations
        self.projectors_protein = MoBYMLP(
            in_dim=proj_dim, 
            inner_dim=protein_inner_dim, 
            out_dim=protein_out_dim,
            num_layers=protein_num_projector
        )

        self.projectors_residue = MoBYMLP(
            in_dim=proj_dim, 
            inner_dim=residue_inner_dim, 
            out_dim=residue_out_dim,
            num_layers=residue_num_projector
        )
        
        # Fine-tuning control
        if hasattr(configs.model.Geom2vec_tematt_encoder, "fine_tuning") and not configs.model.Geom2vec_tematt_encoder.fine_tuning.enable:
            for param in self.input_projection.parameters():
                param.requires_grad = False
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.attention.parameters():
                param.requires_grad = False

        if hasattr(configs.model.Geom2vec_tematt_encoder, "fine_tuning_projct") and not configs.model.Geom2vec_tematt_encoder.fine_tuning_projct.enable:
            for param in self.projectors_protein.parameters():
                param.requires_grad = False
            for param in self.projectors_residue.parameters():
                param.requires_grad = False

    def forward(self, graph, return_logits=False, return_embedding=False, return_attention=False):
        """
        Forward pass through BiLSTM with attention
        
        Args:
            graph: Tuple of (res_rep, batch_idx)
            return_logits: Whether to return logits (placeholder)
            return_embedding: Whether to return intermediate embeddings
            return_attention: Whether to return attention weights
        """
        residue_feature_list=[]
        for gt in graph:
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=None)
            graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
            residue_feature_list.append(residue_feature_embedding)

        stacked = torch.stack(residue_feature_list, dim=0)
        # Step 2: Permute to [M, T, dim]
        res_rep = stacked.permute(1, 0, 2)


        if return_logits:
            print("Logits mode not implemented for BiLSTM version")
            return None
        
        # res_rep = graph[0]  # [sum_n_residue, T, 512]
        batch_idx = gt.batch  # [sum_n_residue]
        
        # Input projection
        x = self.input_projection(res_rep)  # [sum_n_residue, T, hidden_dim]
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)  # Apply batch norm across feature dim
        x = self.dropout(x)
        
        # BiLSTM encoding
        lstm_output, (hidden, cell) = self.lstm(x)  # [sum_n_residue, T, lstm_output_dim]
        
        # Apply attention mechanism
        attended_output, attention_weights = self.attention(lstm_output)  # [sum_n_residue, lstm_output_dim]
        
        # Residue-level representation (after attention)
        res_rep_final = attended_output  # [sum_n_residue, lstm_output_dim]
        
        # Protein-level representation (graph pooling)
        protein_level_rep = scatter_mean(res_rep_final, batch_idx, dim=0)  # [B, lstm_output_dim]
        
        # Apply projectors
        graph_feature = self.projectors_protein(protein_level_rep)  # [B, protein_out_dim]
        residue_feature = self.projectors_residue(res_rep_final)  # [sum_n_residue, residue_out_dim]
        
        if return_embedding and return_attention:
            return graph_feature, residue_feature, protein_level_rep, res_rep_final, attention_weights
        elif return_embedding:
            return graph_feature, residue_feature, protein_level_rep, res_rep_final
        elif return_attention:
            return graph_feature, residue_feature, attention_weights
        else:
            return graph_feature, residue_feature

class HoulsbyAdapter(nn.Module):
    """
    Houlsby-style bottleneck adapter module.
    Architecture: LayerNorm -> Down-project -> Non-linearity -> Up-project -> Residual
    """
    def __init__(self, hidden_size: int, bottleneck_size: int = 64, non_linearity: str = "gelu"):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Non-linearity selection
        if non_linearity == "gelu":
            self.act = nn.GELU()
        elif non_linearity == "relu":
            self.act = nn.ReLU()
        elif non_linearity == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()
        
        # Houlsby initialization: near-identity at start
        self._init_weights()
    
    def _init_weights(self):
        # Initialize down projection with small values
        nn.init.normal_(self.down_proj.weight, std=1e-2)
        nn.init.zeros_(self.down_proj.bias)
        # Initialize up projection to near-zero for identity-like behavior
        nn.init.normal_(self.up_proj.weight, std=1e-2)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x + residual

class VJEPA2LayerWithAdapters(nn.Module):
    """
    Modified VJEPA2Layer with Houlsby adapters.
    
    Original architecture:
        norm1 -> attention -> drop_path -> [hidden_states + attention_output]
        norm2 -> mlp -> [previous_output + mlp_output]
    
    With adapters:
        norm1 -> attention -> drop_path -> [hidden_states + attention_output] -> adapter_attention
        norm2 -> mlp -> [previous_output + mlp_output] -> adapter_mlp
    """
    def __init__(
        self, 
        original_layer,
        bottleneck_size: int = 64, 
        non_linearity: str = "gelu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Extract hidden size from attention query projection
        hidden_size = original_layer.attention.query.in_features
        
        # Copy all original components from VJEPA2Layer
        self.norm1 = original_layer.norm1
        self.attention = original_layer.attention
        self.drop_path = original_layer.drop_path
        self.norm2 = original_layer.norm2
        self.mlp = original_layer.mlp
        
        # Add Houlsby adapters after attention and MLP
        self.adapter_attention = HoulsbyAdapter(
            hidden_size, bottleneck_size, non_linearity
        )
        self.adapter_mlp = HoulsbyAdapter(
            hidden_size, bottleneck_size, non_linearity
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        # head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass matching VJEPA2Layer structure with adapters.
        
        Args:
            hidden_states: Input tensor (batch_size, sequence_length, hidden_size)
            head_mask: Optional attention mask
            output_attentions: Whether to return attention weights
        """
        # Attention block with residual
        attention_output = self.attention(
            self.norm1(hidden_states),
            # head_mask=head_mask,
            output_attentions=output_attentions,
        )
        
        # Handle tuple output from attention
        if isinstance(attention_output, tuple):
            if output_attentions:
                attention_output, attention_weights = attention_output
            else:
                attention_output = attention_output[0]
        
        # Apply drop_path and residual connection
        hidden_states = hidden_states + self.drop_path(attention_output)
        
        # Insert adapter after attention
        hidden_states = self.adapter_attention(hidden_states)
        
        # MLP block with residual
        mlp_output = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + self.drop_path(mlp_output)
        
        # Insert adapter after MLP
        hidden_states = self.adapter_mlp(hidden_states)
        
        if output_attentions:
            return (hidden_states, attention_weights)
        return (hidden_states,)


class VJEPA2WithAdapters(nn.Module):
    """
    V-JEPA2 model with Houlsby adapters for parameter-efficient fine-tuning.
    
    Supports both feature extraction (VJEPA2Model) and classification 
    (VJEPA2ForVideoClassification).
    """
    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        num_labels: Optional[int] = None,
        bottleneck_size: int = 64,
        non_linearity: str = "gelu",
        dropout: float = 0.0,
        freeze_base: bool = True,
        add_classification_head: bool = False
    ):
        super().__init__()
        
        # Load pretrained V-JEPA2
        if add_classification_head and num_labels:
            # self.model = VJEPA2ForVideoClassification.from_pretrained(
            #     model_name,
            #     num_labels=num_labels,
            #     ignore_mismatched_sizes=True
            # )
            self.has_classifier = True
        else:
            self.model = AutoModel.from_pretrained(model_name)
            # self.model = VJEPA2Model.from_pretrained(model_name)
            self.has_classifier = False
            
        self.config = self.model.config
        self.bottleneck_size = bottleneck_size
        
        # Add adapters to all encoder layers
        self._add_adapters_to_encoder(bottleneck_size, non_linearity, dropout)
        
        # Freeze base model parameters
        if freeze_base:
            self._freeze_base_model()
    
    def _add_adapters_to_encoder(
        self, 
        bottleneck_size: int, 
        non_linearity: str,
        dropout: float
    ):
        """
        Replace each VJEPA2Layer with adapter-augmented version.
        
        V-JEPA2 has 24 layers accessed via:
        - VJEPA2Model: model.encoder.layer
        - VJEPA2ForVideoClassification: model.vjepa2.encoder.layer
        """
        # Get encoder based on model type
        if self.has_classifier:
            encoder = self.model.vjepa2.encoder
        else:
            encoder = self.model.encoder
        
        # Replace each layer
        num_layers = len(encoder.layer)
        print(f"Adding adapters to {num_layers} VJEPA2 layers...")
        
        for i in range(num_layers):
            original_layer = encoder.layer[i]
            encoder.layer[i] = VJEPA2LayerWithAdapters(
                original_layer, 
                bottleneck_size, 
                non_linearity, 
                dropout
            )
            
        print(f"✓ Successfully added {num_layers * 2} adapters (2 per layer)")
    
    def _freeze_base_model(self):
        """
        Freeze all parameters except adapters and classification head.
        
        Frozen components:
        - Patch embeddings (Conv3d)
        - All attention components (query, key, value, proj)
        - All MLP components (fc1, fc2)
        - Layer norms (norm1, norm2)
        - Final layernorm
        
        Trainable components:
        - All adapter parameters
        - Classification head (if present)
        """
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            if "adapter" in name or "classifier" in name or "head" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"✓ Frozen {frozen_count} parameter groups")
        print(f"✓ Trainable {trainable_count} parameter groups (adapters + head)")
    
    def unfreeze_layernorm(self):
        """Optionally unfreeze all layer normalization parameters."""
        for name, param in self.model.named_parameters():
            if "norm" in name.lower() or "layernorm" in name.lower():
                param.requires_grad = True
    
    def unfreeze_classifier(self):
        """Unfreeze the classification head."""
        if self.has_classifier:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self):
        """Return count and percentage of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        percentage = 100 * trainable / total if total > 0 else 0
        return trainable, total, percentage
    
    def get_adapter_params(self):
        """Return only adapter parameters."""
        adapter_params = sum(
            p.numel() for name, p in self.named_parameters() 
            if "adapter" in name
        )
        return adapter_params
    
    def forward(
        self, 
        pixel_values_videos: torch.Tensor,
        # labels: Optional[torch.Tensor] = None,
        # output_attentions: bool = False,
        # output_hidden_states: bool = False,
        # return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass through V-JEPA2 with adapters.
        
        Args:
            pixel_values_videos: Video tensor of shape 
                (batch_size, num_frames, num_channels, height, width)
            labels: Optional labels for classification
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dict or tuple
        """
        # return self.model(
        #     pixel_values_videos=pixel_values_videos,
        #     labels=labels,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        return self.model.get_vision_features(pixel_values_videos) #[B, token_len, dim] there is no cls


# =============================================================================
# Utility Functions
# =============================================================================

def create_vjepa2_with_adapters(
    model_name: str = "facebook/vjepa2-vitl-fpc64-256",
    num_labels: Optional[int] = None,
    bottleneck_size: int = 64,
    non_linearity: str = "gelu",
    dropout: float = 0.0,
    add_classification_head: bool = False,
    verbose: bool = True
) -> VJEPA2WithAdapters:
    """
    Factory function to create V-JEPA2 with Houlsby adapters.
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classes for classification
        bottleneck_size: Adapter bottleneck dimension (32, 64, 128, 256)
        non_linearity: Activation function ('gelu', 'relu', 'silu', 'tanh')
        dropout: Dropout rate in adapters (0.0 to 0.2 recommended)
        add_classification_head: Whether to add classification head
        verbose: Print detailed information
    """
    if verbose:
        print("\n" + "="*70)
        print("Creating V-JEPA2 Model with Houlsby Adapters")
        print("="*70)
        print(f"Model: {model_name}")
        print(f"Bottleneck size: {bottleneck_size}")
        print(f"Non-linearity: {non_linearity}")
        print(f"Dropout: {dropout}")
        print(f"Classification head: {add_classification_head}")
        if add_classification_head:
            print(f"Number of labels: {num_labels}")
        print("-"*70)
    
    model = VJEPA2WithAdapters(
        model_name=model_name,
        num_labels=num_labels,
        bottleneck_size=bottleneck_size,
        non_linearity=non_linearity,
        dropout=dropout,
        freeze_base=True,
        add_classification_head=add_classification_head
    )
    
    # if verbose:
    #     print_model_summary(model)
    
    return model

class VJEPA2_tune(nn.Module):  # embedding table is fixed
    def __init__(self,
                 configs,
                 dim_mlp = 768,
                 protein_out_dim=256,
                 protein_inner_dim=4096, 
                 protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(VJEPA2_tune, self).__init__()
        # self.image_processor = VivitImageProcessor.from_pretrained(vivit_pretrain)
        # self.model = VivitModel.from_pretrained(vivit_pretrain, cache_dir=configs.HF_cache_path)

        # dim_mlp = 768
        
        self.projectors_protein = MoBYMLP(in_dim=dim_mlp, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        # self.projectors_residue = MoBYMLP(in_dim=dim_mlp, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
        #                                   num_layers=residue_num_projector)
        
        # self.model = VivitModel.from_pretrained(configs.model.MD_encoder.model_name)
        if configs.model.MD_encoder.fine_tuning.enable:
            self.model = AutoModel.from_pretrained(configs.model.MD_encoder.model_name)
            unfix_last_layer = configs.model.MD_encoder.fine_tuning.unfix_last_layer
            fix_layer_num = self.model.config.num_hidden_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.model.encoder.layer:  # only fine-tune transformer layers,no contact_head and other parameters
                if fix_layer_index < fix_layer_num:
                    for p in layer.parameters():
                    # logging.info('unfix layer')
                        p.requires_grad = False
                    fix_layer_index += 1  # keep these layers frozen
                    continue

                for p in layer.parameters():
                    # logging.info('unfix layer')
                    p.requires_grad = True

        elif configs.model.MD_encoder.adapter_h.enable:
            self.model = create_vjepa2_with_adapters(
                model_name="facebook/vjepa2-vitl-fpc64-256",
                # num_labels=10,  # Your target number of classes
                bottleneck_size=64,  # Adapter bottleneck dimension
                non_linearity="gelu"
            )
        self.configs = configs

        # if hasattr(configs.model.MD_encoder, "fine_tuning") and not configs.model.MD_encoder.fine_tuning.enable:
        #     for name, param in self.model.named_parameters():
        #         param.requires_grad = False

        if hasattr(configs.model.MD_encoder, "fine_tuning_projct") and not configs.model.MD_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False
            
            # for name, param in self.projectors_residue.named_parameters():
            #     param.requires_grad = False

    def forward(self, x, return_logits=False, return_embedding=False): # x shape [batch, 32, 3, 224, 224]
        #outputs = self.model(x)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            # prediction_scores = outputs["logits"]
            # return prediction_scores
            print("print something")
        else:
            # last_hidden_states = outputs.last_hidden_state # [batch, 3137, 768]
            # cls_representation = last_hidden_states[:, 0, :] # [batch, 768]
            if self.configs.model.MD_encoder.fine_tuning.enable:
                output = self.model.get_vision_features(x)
            elif self.configs.model.MD_encoder.adapter_h.enable:
                output = self.model(x) #[batch, token_len=8192, dim=1024]. no cls
            # last_hidden_states = vivit_output.last_hidden_state #[batch, 3137, 768]
            representation = torch.mean(output, dim=1) #[batch, 1024]

            graph_feature = self.projectors_protein(representation)

        if return_embedding:
            return graph_feature, representation
        else:
            return graph_feature

class VivitLayerWithAdapters(nn.Module):
    """
    Modified ViViT layer with Houlsby adapters inserted after attention and FFN.
    """
    def __init__(self, original_layer: VivitLayer, bottleneck_size: int = 64, non_linearity: str = "gelu"):
        super().__init__()
        hidden_size = original_layer.attention.attention.all_head_size
        
        # Copy original components
        self.attention = original_layer.attention
        self.intermediate = original_layer.intermediate
        self.output = original_layer.output
        self.layernorm_before = original_layer.layernorm_before
        self.layernorm_after = original_layer.layernorm_after
        
        # Add adapters after attention and FFN
        self.adapter_attention = HoulsbyAdapter(hidden_size, bottleneck_size, non_linearity)
        self.adapter_ffn = HoulsbyAdapter(hidden_size, bottleneck_size, non_linearity)
    
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        # Self-attention with pre-norm
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        
        # First residual + adapter after attention
        hidden_states = attention_output + hidden_states
        hidden_states = self.adapter_attention(hidden_states)
        
        # FFN with pre-norm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        
        # Adapter after FFN
        layer_output = self.adapter_ffn(layer_output)
        
        outputs = (layer_output,) + outputs
        return outputs


class VivitWithAdapters(nn.Module):
    """
    ViViT model with Houlsby adapters for parameter-efficient fine-tuning.
    Freezes the base model and only trains adapter parameters.
    """
    def __init__(
        self,
        model_name: str = "google/vivit-b-16x2-kinetics400",
        # num_labels: int = 400,
        bottleneck_size: int = 64,
        non_linearity: str = "gelu",
        freeze_base: bool = True,
        num_layers: int =4
    ):
        super().__init__()
        
        # Load pretrained ViViT
        # self.vivit = VivitForVideoClassification.from_pretrained(
        self.vivit = VivitModel.from_pretrained(
            model_name,
            # num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.config = self.vivit.config
        self.bottleneck_size = bottleneck_size
        
        # Replace layers with adapter-augmented versions
        self._add_adapters(bottleneck_size, non_linearity, num_layers)
        
        # Freeze base model parameters
        if freeze_base:
            self._freeze_base_model()
    
    def _add_adapters(self, bottleneck_size: int, non_linearity: str, num_layers: int):
        """Replace each VivitLayer with adapter-augmented version."""
        fix_layer_num = self.vivit.config.num_hidden_layers - num_layers
        for i, layer in enumerate(self.vivit.encoder.layer):
            if i < fix_layer_num:
                continue
            else:
                self.vivit.encoder.layer[i] = VivitLayerWithAdapters(
                    layer, bottleneck_size, non_linearity
                )
    
    def _freeze_base_model(self):
        """Freeze all parameters except adapters and classification head."""
        for name, param in self.vivit.named_parameters():
            if "adapter" not in name and "classifier" not in name:
                param.requires_grad = False
        for p in self.vivit.layernorm.parameters():
            p.requires_grad = True
    
    def unfreeze_classifier(self):
        """Optionally unfreeze the classification head."""
        for param in self.vivit.classifier.parameters():
            param.requires_grad = True
        
    
    def get_trainable_params(self):
        """Return count and percentage of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total, 100 * trainable / total
    
    def forward(self, pixel_values, labels=None, **kwargs):
        # return self.vivit(pixel_values=pixel_values, labels=labels, **kwargs)
        return self.vivit(pixel_values=pixel_values, **kwargs)


# =============================================================================
# Training utilities
# =============================================================================

def create_adapter_vivit(
    model_name: str = "google/vivit-b-16x2-kinetics400",
    num_labels: int = 10,
    bottleneck_size: int = 64,
    non_linearity: str = "gelu",
    num_layers: int = 4
) -> VivitWithAdapters:
    """Factory function to create ViViT with adapters."""
    model = VivitWithAdapters(
        model_name=model_name,
        # num_labels=num_labels,
        bottleneck_size=bottleneck_size,
        non_linearity=non_linearity,
        freeze_base=True,
        num_layers=num_layers
    )
    trainable, total, pct = model.get_trainable_params()
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    return model

class VIVIT_tune(nn.Module):  # embedding table is fixed
    def __init__(self, vivit_pretrain, logging,
                 accelerator,
                 configs,
                 dim_mlp = 768,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(VIVIT_tune, self).__init__()
        # self.image_processor = VivitImageProcessor.from_pretrained(vivit_pretrain)
        # self.model = VivitModel.from_pretrained(vivit_pretrain, cache_dir=configs.HF_cache_path)

        # dim_mlp = 768
        
        self.projectors_protein = MoBYMLP(in_dim=dim_mlp, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        # self.projectors_residue = MoBYMLP(in_dim=dim_mlp, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
        #                                   num_layers=residue_num_projector)
        
        # self.model = VivitModel.from_pretrained(configs.model.MD_encoder.model_name)
        if configs.model.MD_encoder.fine_tuning.enable:
            self.model = VivitModel.from_pretrained(configs.model.MD_encoder.model_name)
            unfix_last_layer = configs.model.MD_encoder.fine_tuning.unfix_last_layer
            fix_layer_num = self.model.config.num_hidden_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.model.encoder.layer:  # only fine-tune transformer layers,no contact_head and other parameters
                if fix_layer_index < fix_layer_num:
                    for p in layer.parameters():
                    # logging.info('unfix layer')
                        p.requires_grad = False
                    fix_layer_index += 1  # keep these layers frozen
                    continue

                for p in layer.parameters():
                    # logging.info('unfix layer')
                    p.requires_grad = True

        elif configs.model.MD_encoder.adapter_h.enable:
            self.model = create_adapter_vivit(
                model_name="google/vivit-b-16x2-kinetics400",
                # num_labels=10,  # Your target number of classes
                bottleneck_size=128,  # Adapter bottleneck dimension
                non_linearity="gelu",
                num_layers=configs.model.MD_encoder.adapter_h.num_end_adapter_layers
            )
        elif configs.model.MD_encoder.lora.enable:
            self.model = VivitModel.from_pretrained(configs.model.MD_encoder.model_name)
            lora_targets =  ["attention.attention.query", "attention.attention.key",
                              "attention.attention.value","attention.output.dense"]
            target_modules=[]
            if configs.model.MD_encoder.lora.num_end_lora > 0:
                start_layer_idx = np.max([self.model.config.num_hidden_layers - configs.model.MD_encoder.lora.num_end_lora, 0])
                for idx in range(start_layer_idx, self.model.config.num_hidden_layers):
                    for layer_name in lora_targets:
                        target_modules.append(f"encoder.layer.{idx}.{layer_name}")
                
            peft_config = LoraConfig(
                inference_mode=False,
                r=configs.model.MD_encoder.lora.r,
                lora_alpha=configs.model.MD_encoder.lora.alpha,
                target_modules=target_modules,
                lora_dropout=configs.model.MD_encoder.lora.dropout,
                bias="none",
                # modules_to_save=modules_to_save
            )
            self.peft_model = get_peft_model(self.model, peft_config)

        # if hasattr(configs.model.MD_encoder, "fine_tuning") and not configs.model.MD_encoder.fine_tuning.enable:
        #     for name, param in self.model.named_parameters():
        #         param.requires_grad = False

        if hasattr(configs.model.MD_encoder, "fine_tuning_projct") and not configs.model.MD_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False
            
            # for name, param in self.projectors_residue.named_parameters():
            #     param.requires_grad = False

    def forward(self, x, return_logits=False, return_embedding=False): # x shape [batch, 32, 3, 224, 224]
        #outputs = self.model(x)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            # prediction_scores = outputs["logits"]
            # return prediction_scores
            print("print something")
        else:
            # last_hidden_states = outputs.last_hidden_state # [batch, 3137, 768]
            # cls_representation = last_hidden_states[:, 0, :] # [batch, 768]
            vivit_output = self.model(x) # 
            last_hidden_states = vivit_output.last_hidden_state #[batch, 3137, 768]
            cls_representation = last_hidden_states[:, 0, :] #[1, 768]

            graph_feature = self.projectors_protein(cls_representation)

        if return_embedding:
            return graph_feature, cls_representation
        else:
            return graph_feature

class VIVIT(nn.Module):  # embedding table is fixed
    def __init__(self, vivit_pretrain, logging,
                 accelerator,
                 configs,
                 dim_mlp = 768,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(VIVIT, self).__init__()
        # self.image_processor = VivitImageProcessor.from_pretrained(vivit_pretrain)
        # self.model = VivitModel.from_pretrained(vivit_pretrain, cache_dir=configs.HF_cache_path)

        # dim_mlp = 768
        
        self.projectors_protein = MoBYMLP(in_dim=dim_mlp, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        # self.projectors_residue = MoBYMLP(in_dim=dim_mlp, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
        #                                   num_layers=residue_num_projector)
        

        # if hasattr(configs.model.MD_encoder, "fine_tuning") and not configs.model.MD_encoder.fine_tuning.enable:
        #     for name, param in self.model.named_parameters():
        #         param.requires_grad = False

        if hasattr(configs.model.MD_encoder, "fine_tuning_projct") and not configs.model.MD_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False
            
            # for name, param in self.projectors_residue.named_parameters():
            #     param.requires_grad = False

    def forward(self, x, return_logits=False, return_embedding=False): # x shape [batch, 768]
        #outputs = self.model(x)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            # prediction_scores = outputs["logits"]
            # return prediction_scores
            print("print something")
        else:
            # last_hidden_states = outputs.last_hidden_state # [batch, 3137, 768]
            # cls_representation = last_hidden_states[:, 0, :] # [batch, 768]
            graph_feature = self.projectors_protein(x)

        if return_embedding:
            return graph_feature, x
        else:
            return graph_feature

class Geom2vec_tematt(nn.Module):  
    def __init__(self, 
                 configs,
                 input_dim=512, 
                 hidden_dim=128, 
                 output_dim=32,
                 temporal_dim=100,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(Geom2vec_tematt, self).__init__()
        # self.image_processor = VivitImageProcessor.from_pretrained(vivit_pretrain)
        # self.model = VivitModel.from_pretrained(vivit_pretrain, cache_dir=configs.HF_cache_path)

        # First conv: maintain temporal resolution
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=7,
            stride=1,
            padding=3
        )
        
        # Progressive pooling: 1000 -> 500
        # self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Second conv: still maintain some resolution
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=5,
            stride=1,
            padding=2
        )
        
        # Final pooling: 500 -> 100
        # self.pool2 = nn.AvgPool1d(kernel_size=5, stride=5)
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.1)
        
        proj_dim= temporal_dim * output_dim
        self.projectors_protein = MoBYMLP(in_dim=proj_dim, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        self.projectors_residue = MoBYMLP(in_dim=proj_dim, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
                                          num_layers=residue_num_projector)
        

        if hasattr(configs.model.Geom2vec_tematt_encoder, "fine_tuning") and not configs.model.Geom2vec_tematt_encoder.fine_tuning.enable:
            for name, param in self.conv1.named_parameters():
                param.requires_grad = False
            for name, param in self.conv2.named_parameters():
                param.requires_grad = False

        if hasattr(configs.model.Geom2vec_tematt_encoder, "fine_tuning_projct") and not configs.model.Geom2vec_tematt_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False
            
            for name, param in self.projectors_residue.named_parameters():
                param.requires_grad = False

    def forward(self, graph, return_logits=False, return_embedding=False): #res_rep (flat) [sum_n_residue, T, 512]
        #outputs = self.model(x)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            # prediction_scores = outputs["logits"]
            # return prediction_scores
            print("print something")
        else:
            res_rep = graph[0]
            batch_idx = graph[1]
            x = res_rep.transpose(1, 2)  # [sum_n_residue, dim, T]
            
            # First block: feature extraction without downsampling
            x = self.conv1(x)  # [sum_n_residue, dim, T]
            x = self.batch_norm1(x)
            x = self.relu(x)
            x = self.dropout(x)
            # x = self.pool1(x)  # [sum_n_residue, dim, T]
            
            # Second block: more feature extraction
            x = self.conv2(x)  # [sum_n_residue, out_dim, T]
            x = self.batch_norm2(x)
            x = self.relu(x)
            # x = self.pool2(x)  # [sum_n_residue, out_dim, tem_dim]
            
            res_rep = x.transpose(1, 2)  # [sum_n_residue, tem_dim, out_dim]

            # Step 3: Flatten [10, 64] → [640]
            res_rep = res_rep.reshape(res_rep.size(0), -1)  # [sum_n_residue, tem_dim * out_dim]
            
            # Step 4: Graph-style pooling to protein level: [B, 640]
            protein_level_rep = scatter_mean(res_rep, batch_idx, dim=0)  # [B, 640]
            # last_hidden_states = outputs.last_hidden_state # [batch, 3137, 768]
            # cls_representation = last_hidden_states[:, 0, :] # [batch, 768]
            graph_feature = self.projectors_protein(protein_level_rep)
            residue_feature = self.projectors_residue(res_rep)

        if return_embedding:
            return graph_feature, residue_feature, protein_level_rep, res_rep
        else:
            return graph_feature, residue_feature

class Geom2vec(nn.Module):  
    def __init__(self, 
                 configs,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(Geom2vec, self).__init__()
        # self.image_processor = VivitImageProcessor.from_pretrained(vivit_pretrain)
        # self.model = VivitModel.from_pretrained(vivit_pretrain, cache_dir=configs.HF_cache_path)

        dim_res = 512
        dim_pro = 1024
        
        self.projectors_protein = MoBYMLP(in_dim=dim_pro, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        self.projectors_residue = MoBYMLP(in_dim=dim_res, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
                                          num_layers=residue_num_projector)
        

        # if hasattr(configs.model.MD_encoder, "fine_tuning") and not configs.model.MD_encoder.fine_tuning.enable:
        #     for name, param in self.model.named_parameters():
        #         param.requires_grad = False

        if hasattr(configs.model.Geom2vec_encoder, "fine_tuning_projct") and not configs.model.Geom2vec_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False
            
            for name, param in self.projectors_residue.named_parameters():
                param.requires_grad = False

    def forward(self, graph, return_logits=False, return_embedding=False): #prot_vec [batch_size, 1024], res_rep (flat) [sum_n_residue, 512]
        #outputs = self.model(x)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            # prediction_scores = outputs["logits"]
            # return prediction_scores
            print("print something")
        else:
            res_rep = graph[0]
            prot_vec = graph[1]
            # last_hidden_states = outputs.last_hidden_state # [batch, 3137, 768]
            # cls_representation = last_hidden_states[:, 0, :] # [batch, 768]
            graph_feature = self.projectors_protein(prot_vec)
            residue_feature = self.projectors_residue(res_rep)

        if return_embedding:
            return graph_feature, residue_feature, prot_vec, res_rep
        else:
            return graph_feature, residue_feature

class TemporalAttention(nn.Module):
    """Attention mechanism to weight temporal features"""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        # Compute attention weights
        attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, hidden_dim]
        
        return attended_output, attention_weights

class Geom2vec_temlstmatt(nn.Module):  
    def __init__(self, 
                 configs,
                 input_dim=512, 
                 hidden_dim=128, 
                 output_dim=32,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, 
                 protein_num_projector=2,
                 lstm_layers=2,
                 dropout_rate=0.1):
        """
        BiLSTM-based temporal encoder with attention mechanism
        
        Args:
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate for LSTM and attention
        """
        super(Geom2vec_temlstmatt, self).__init__()
        
        # Input projection to reduce dimensionality before LSTM
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=output_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # LSTM output dimension (bidirectional doubles the size)
        lstm_output_dim = output_dim * 2
        
        # Attention mechanism
        self.attention = TemporalAttention(lstm_output_dim)
        
        # Batch normalization and dropout
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final projection dimension after attention
        proj_dim = lstm_output_dim
        
        # Projectors for protein and residue level representations
        self.projectors_protein = MoBYMLP(
            in_dim=proj_dim, 
            inner_dim=protein_inner_dim, 
            out_dim=protein_out_dim,
            num_layers=protein_num_projector
        )

        self.projectors_residue = MoBYMLP(
            in_dim=proj_dim, 
            inner_dim=residue_inner_dim, 
            out_dim=residue_out_dim,
            num_layers=residue_num_projector
        )
        
        # Fine-tuning control
        if hasattr(configs.model.Geom2vec_tematt_encoder, "fine_tuning") and not configs.model.Geom2vec_tematt_encoder.fine_tuning.enable:
            for param in self.input_projection.parameters():
                param.requires_grad = False
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.attention.parameters():
                param.requires_grad = False

        if hasattr(configs.model.Geom2vec_tematt_encoder, "fine_tuning_projct") and not configs.model.Geom2vec_tematt_encoder.fine_tuning_projct.enable:
            for param in self.projectors_protein.parameters():
                param.requires_grad = False
            for param in self.projectors_residue.parameters():
                param.requires_grad = False

    def forward(self, graph, return_logits=False, return_embedding=False, return_attention=False):
        """
        Forward pass through BiLSTM with attention
        
        Args:
            graph: Tuple of (res_rep, batch_idx)
            return_logits: Whether to return logits (placeholder)
            return_embedding: Whether to return intermediate embeddings
            return_attention: Whether to return attention weights
        """
        if return_logits:
            print("Logits mode not implemented for BiLSTM version")
            return None
        
        res_rep = graph[0]  # [sum_n_residue, T, 512]
        batch_idx = graph[1]  # [sum_n_residue]
        
        # Input projection
        x = self.input_projection(res_rep)  # [sum_n_residue, T, hidden_dim]
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)  # Apply batch norm across feature dim
        x = self.dropout(x)
        
        # BiLSTM encoding
        lstm_output, (hidden, cell) = self.lstm(x)  # [sum_n_residue, T, lstm_output_dim]
        
        # Apply attention mechanism
        attended_output, attention_weights = self.attention(lstm_output)  # [sum_n_residue, lstm_output_dim]
        
        # Residue-level representation (after attention)
        res_rep_final = attended_output  # [sum_n_residue, lstm_output_dim]
        
        # Protein-level representation (graph pooling)
        protein_level_rep = scatter_mean(res_rep_final, batch_idx, dim=0)  # [B, lstm_output_dim]
        
        # Apply projectors
        graph_feature = self.projectors_protein(protein_level_rep)  # [B, protein_out_dim]
        residue_feature = self.projectors_residue(res_rep_final)  # [sum_n_residue, residue_out_dim]
        
        if return_embedding and return_attention:
            return graph_feature, residue_feature, protein_level_rep, res_rep_final, attention_weights
        elif return_embedding:
            return graph_feature, residue_feature, protein_level_rep, res_rep_final
        elif return_attention:
            return graph_feature, residue_feature, attention_weights
        else:
            return graph_feature, residue_feature

class EdgeGNNLayer(MessagePassing):
    """
    Custom GNN layer that incorporates edge features in message passing
    """
    def __init__(self, node_in_dim: int, edge_dim: int, node_out_dim: int):
        super(EdgeGNNLayer, self).__init__(aggr='mean')
        
        # Node transformation
        self.node_lin = nn.Linear(node_in_dim, node_out_dim)
        
        # Edge transformation 
        self.edge_lin = nn.Linear(edge_dim, node_out_dim)
        
        # Message transformation
        self.message_lin = nn.Linear(node_out_dim * 2 + edge_dim, node_out_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(node_out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        x = self.node_lin(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Add residual connection and apply layer norm
        out = self.layer_norm(out + x)
        out = self.dropout(out)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features [num_edges, node_dim]
        # x_j: source node features [num_edges, node_dim] 
        # edge_attr: edge features [num_edges, edge_dim]
        
        # Concatenate source node, target node, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Transform concatenated features
        message = self.message_lin(msg_input)
        message = F.relu(message)
        
        return message

class DCCM_GNN(nn.Module):
    """
    Unified Graph Neural Network for processing DCCM graphs with edge features.
    Handles both single graphs and batched graphs automatically.
    """
    def __init__(self, 
                 configs,
                 node_input_dim: int = 10001,
                 edge_dim: int = 2,
                 hidden_dim: int = 512,
                 output_dim: int = 100,
                 num_layers: int = 3,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096,
                 protein_out_dim=256,
                 protein_num_projector=2,
                 pooling: str = 'mean'):
        super(DCCM_GNN, self).__init__()
        
        
        
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EdgeGNNLayer(
                node_in_dim=hidden_dim,
                edge_dim=edge_dim,
                node_out_dim=hidden_dim
            )
            self.gnn_layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.projectors_protein = MoBYMLP(in_dim=output_dim, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        self.projectors_residue = MoBYMLP(in_dim=output_dim, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
                                          num_layers=residue_num_projector)
        
        if hasattr(configs.model.DCCM_GNN_encoder, "fine_tuning") and not configs.model.DCCM_GNN_encoder.fine_tuning.enable:
            for name, param in self.input_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.gnn_layers.named_parameters():
                param.requires_grad = False
            for name, param in self.output_proj.named_parameters():
                param.requires_grad = False

        if hasattr(configs.model.DCCM_GNN_encoder, "fine_tuning_projct") and not configs.model.DCCM_GNN_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False
            
            for name, param in self.projectors_residue.named_parameters():
                param.requires_grad = False
        # Pooling function (for graph-level representations)
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling is not None:
            raise ValueError(f"Unsupported pooling: {pooling}")
        
    def forward(self, data, return_embedding=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Project input features
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
        
        # Final output projection
        residue_emb = self.output_proj(x) #[batch_size * residue_num, dim]
        
        # Apply pooling if specified (for graph-level output)
        if self.pooling is not None:
            # Check if we have batch information (batched graphs)
            if hasattr(data, 'batch'):
                graph_emb = self.pool(residue_emb, data.batch)  # [batch_size, output_dim]
            else:
                # Single graph - pool all nodes
                batch = torch.zeros(residue_emb.size(0), dtype=torch.long, device=x.device)
                graph_emb = self.pool(residue_emb, batch)  # [1, output_dim]
        
        graph_feature = self.projectors_protein(graph_emb)
        residue_feature = self.projectors_residue(residue_emb)
        if return_embedding:
            return graph_feature, residue_feature, graph_emb, residue_emb
        else:
            return graph_feature, residue_feature

class ESM2(nn.Module):  # embedding table is fixed
    def __init__(self, esm2_pretrain, logging,
                 accelerator,
                 configs,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(ESM2, self).__init__()
        if configs.model.esm_encoder.adapter_h.enable:
            if accelerator.is_main_process:
                logging.info("use adapter H")
            # num_end_adapter_layers = configs.model.esm_encoder.adapter_h.num_end_adapter_layers
            adapter_args = configs.model.esm_encoder.adapter_h
            esm2_dict = {
                         "esm2_t36_3B_UR50D": esm_adapterH.pretrained.esm2_t36_3B_UR50D(adapter_args),  # 36 layers embedding=2560
                         "esm2_t33_650M_UR50D": esm_adapterH.pretrained.esm2_t33_650M_UR50D(adapter_args),
                         # 33 layers embedding=1280
                         "esm2_t30_150M_UR50D": esm_adapterH.pretrained.esm2_t30_150M_UR50D(adapter_args),
                         # 30 layers embedding=640
                         "esm2_t12_35M_UR50D": esm_adapterH.pretrained.esm2_t12_35M_UR50D(adapter_args),
                         # 12 layers embedding=480
                         "esm2_t6_8M_UR50D": esm_adapterH.pretrained.esm2_t6_8M_UR50D(adapter_args),
                         # 6 layers embedding = 320
                         }
        else:
            esm2_dict = {
                         "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D(),  # 36 layers embedding=2560
                         "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D(),  # 33 layers embedding=1280
                         "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D(),  # 30 layers embedding=640
                         "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D(),  # 12 layers embedding=480
                         "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D(),  # 6 layers embedding = 320
                         }

        # if esm2_pretrain_local is None:
        self.esm2, self.alphabet = esm2_dict[esm2_pretrain]  # alphabet.all_toks
        # else:
        #     print("load esm2 model from local dir")
        #     self.esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)

        self.num_layers = self.esm2.num_layers
        for p in self.esm2.parameters():  # frozen all parameters first
            p.requires_grad = False
            
        if configs.model.esm_encoder.adapter_h.enable:
            if not isinstance(configs.model.esm_encoder.adapter_h.freeze_adapter_layers, list):
                configs.model.esm_encoder.adapter_h.freeze_adapter_layers = [configs.model.esm_encoder.adapter_h.freeze_adapter_layers]
            for adapter_idx, value in enumerate(configs.model.esm_encoder.adapter_h.freeze_adapter_layers):
                if not value:
                    for name, param in self.esm2.named_parameters():
                        adapter_name = f"adapter_{adapter_idx}"
                        if adapter_name in name:
                            param.requires_grad = True

        if configs.model.esm_encoder.lora.enable:
            if accelerator.is_main_process:
                logging.info('use lora for esm v2')
            if configs.model.esm_encoder.lora.resume.enable:
                self.esm2 = load_and_add_lora_checkpoint(self.esm2, configs.model.esm_encoder.lora.resume)
            else:
                #target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
                lora_targets =  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj","self_attn.out_proj"]
                target_modules=[]
                if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                    start_layer_idx = np.max([self.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                    for idx in range(start_layer_idx, self.num_layers):
                        for layer_name in lora_targets:
                            target_modules.append(f"layers.{idx}.{layer_name}")
                    
                peft_config = LoraConfig(
                    inference_mode=False,
                    r=configs.model.esm_encoder.lora.r,
                    lora_alpha=configs.model.esm_encoder.lora.alpha,
                    target_modules=target_modules,
                    lora_dropout=configs.model.esm_encoder.lora.dropout,
                    bias="none",
                    # modules_to_save=modules_to_save
                )
                self.peft_model = get_peft_model(self.esm2, peft_config)
        elif configs.model.esm_encoder.fine_tuning.enable:
            if accelerator.is_main_process:
                logging.info('fine-tune esm v2')
            unfix_last_layer = configs.model.esm_encoder.fine_tuning.unfix_last_layer
            fix_layer_num = self.num_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.esm2.layers:  # only fine-tune transformer layers,no contact_head and other parameters
                if fix_layer_index < fix_layer_num:
                    fix_layer_index += 1  # keep these layers frozen
                    continue

                for p in layer.parameters():
                    # logging.info('unfix layer')
                    p.requires_grad = True

            if unfix_last_layer != 0:  # if need fine-tune last layer, the emb_layer_norm_after for last representation should updated
                for p in self.esm2.emb_layer_norm_after.parameters():
                    p.requires_grad = True


        self.projectors_residue = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=residue_inner_dim,
                                          num_layers=residue_num_projector,
                                          out_dim=residue_out_dim)

        self.projectors_protein = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=protein_inner_dim,
                                          num_layers=protein_num_projector,
                                          out_dim=protein_out_dim)

    def forward(self, x, return_logits=False, return_embedding=False):
        outputs = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            prediction_scores = outputs["logits"]
            return prediction_scores
        else:
            residue_feature = outputs['representations'][self.num_layers]
            # residue_dim = residue_feature.shape #[batch,L,D]
            # average pooling but remove padding tokens
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature_embedding = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom  # remove padding
            # print("size of graph_feature_embedding is :")
            # print(len(graph_feature_embedding))
            graph_feature = self.projectors_protein(graph_feature_embedding)  # the cls and eos token is included
            mask = ((x != self.alphabet.padding_idx) & (x != self.alphabet.cls_idx) & (
                    x != self.alphabet.eos_idx))  # use this in v2 training
            residue_feature_embedding = residue_feature[mask]
            residue_feature = self.projectors_residue(residue_feature_embedding)
            # residue_feature = self.projectors_residue(residue_feature.view(-1, residue_dim[-1])).view(residue_dim[0],residue_dim[1],-1)
            if return_embedding:
                return graph_feature, residue_feature, graph_feature_embedding, residue_feature_embedding
            else:
                return graph_feature, residue_feature


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) + 1) / 2
        return adj


def reset_current_samelen_batch():
    batch_seq_samelen = []
    images_samelen = []
    return batch_seq_samelen, images_samelen


def check_samelen_batch(batch_size, batch_seq_samelen):
    if len(batch_seq_samelen) == batch_size:
        return True
    else:
        return False


def reset_current_mixlen_batch():
    batch_seq_mixlen = []
    images_mixlen = []
    return batch_seq_mixlen, images_mixlen


def check_mixlen_batch(batch_size, batch_seq_mixlen):
    if len(batch_seq_mixlen) == batch_size:
        return True
    else:
        return False


class MaskedLMDataCollator:
    """Data collator for masked language modeling.
    
    The idea is based on the implementation of DataCollatorForLanguageModeling at
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L751C7-L782
    """
    
    def __init__(self, batch_converter, mlm_probability=0.15):
        """_summary_

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
        """
        self.mlm_probability = mlm_probability
        self.special_token_indices = [batch_converter.alphabet.cls_idx, 
                                batch_converter.alphabet.padding_idx, 
                                batch_converter.alphabet.eos_idx,  
                                batch_converter.alphabet.unk_idx, 
                                batch_converter.alphabet.mask_idx]
        self.vocab_size = batch_converter.alphabet.all_toks.__len__()
        self.mask_idx = batch_converter.alphabet.mask_idx
        

        
    def get_special_tokens_mask(self, tokens):
        return [1 if token in self.special_token_indices else 0 for token in tokens]
    
    def mask_tokens(self, batch_tokens):
        """make a masked input and label from batch_tokens.

        Args:
            batch_tokens (tensor): tensor of batch tokens
            batch_converter (tensor): batch converter from ESM-2.

        Returns:
            inputs: inputs with masked
            labels: labels for masked tokens
        """
        ## mask tokens
        inputs = batch_tokens.clone().to(batch_tokens.device)
        labels = batch_tokens.clone().to(batch_tokens.device)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx
        
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size,
                                     labels.shape, dtype=torch.long).to(batch_tokens.device)
        
        #print(indices_random)
        #print(random_words)
        #print(inputs)
        
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class SimCLR(nn.Module):
    def __init__(self, model_seq, model_x, configs):
        super(SimCLR, self).__init__()
        self.model_seq = model_seq
        self.model_x = model_x
        # if configs.model.X_module == "structure":
            # self.model_struct = model_x
        # elif configs.model.X_module == "MD":
            # self.model_MD = model_x
        self.temperature = configs.train_settings.temperature
        self.n_views = configs.train_settings.n_views
        self.configs = configs

    def forward(self, graph=None, batch_tokens=None, mode=False, return_embedding=False,return_logits=False):
        if mode == 'sequence':
            return self.model_seq(batch_tokens, return_embedding=return_embedding,return_logits=return_logits)
        elif mode == 'structure':
            return self.model_x(graph, return_embedding=return_embedding)
        elif mode =="MD" or mode =='vivit' or mode == 'MD_tune' or \
            mode == "VJEPA2_tune":
            return self.model_x(graph)
        elif mode == 'DCCM_GNN':
            return self.model_x(graph, return_embedding=return_embedding)
        elif mode == 'Geom2vec':
            return self.model_x(graph[0], graph[1]) # graph = [res_rep, prot_vec]
        elif mode == 'Geom2vec_tematt' or mode == 'Geom2vec_temlstmatt':
            return self.model_x(graph[0], graph[1]) # graph = [res_rep, batch_ind]
        else:
            if self.configs.model.X_module == "structure":
                features_seq, residue_seq = self.model_seq(batch_tokens)
                features_struct, residue_struct = self.model_x(graph)
                return features_struct, residue_struct, features_seq, residue_seq
            if self.configs.model.X_module == "MD" or \
                self.configs.model.X_module == "vivit" or \
                    self.configs.model.X_module == "MD_tune" or \
                        self.configs.model.X_module == "VJEPA2_tune":
                
                features_MD = self.model_x(graph)
                if return_embedding:
                    features_seq, residue_seq, graph_feature_embedding, residue_feature_embedding = self.model_seq(batch_tokens, return_embedding=return_embedding)
                    return features_MD, features_seq, residue_seq, graph_feature_embedding, residue_feature_embedding
                else:
                    features_seq, residue_seq = self.model_seq(batch_tokens)
                    return features_MD, features_seq, residue_seq
            if self.configs.model.X_module == "DCCM_GNN":
                features_seq, residue_seq = self.model_seq(batch_tokens)
                protein_MD_feature, residue_MD_feature = self.model_x(graph)
                return protein_MD_feature, residue_MD_feature, features_seq, residue_seq
            if self.configs.model.X_module == "Geom2vec":
                features_seq, residue_seq = self.model_seq(batch_tokens)
                protein_MD_feature, residue_MD_feature = self.model_x(graph) # graph = [res_rep, prot_vec]
                return protein_MD_feature, residue_MD_feature, features_seq, residue_seq
            if self.configs.model.X_module == "Geom2vec_tematt":
                features_seq, residue_seq = self.model_seq(batch_tokens)
                protein_MD_feature, residue_MD_feature = self.model_x(graph) # graph = [res_rep, batch_ind]
                return protein_MD_feature, residue_MD_feature, features_seq, residue_seq
            if self.configs.model.X_module == "Geom2vec_temlstmatt":
                features_seq, residue_seq = self.model_seq(batch_tokens)
                protein_MD_feature, residue_MD_feature = self.model_x(graph) # graph = [res_rep, batch_ind]
                return protein_MD_feature, residue_MD_feature, features_seq, residue_seq
        
    # def forward_structure(self, graph):
    #     features_struct, residue_struct = self.model_struct(graph)
    #     return features_struct, residue_struct

    def forward_sequence(self, batch_tokens):
        features_seq, residue_seq = self.model_seq(batch_tokens)
        return features_seq, residue_seq
    
    def forward_x(self, graph):
        if self.configs.model.X_module == "structure":
            features_struct, residue_struct = self.model_x(graph)
            return features_struct, residue_struct
        if self.configs.model.X_module == "MD" or \
            self.configs.model.X_module == "MD_tune" or \
                self.configs.model.X_module == "VJEPA2_tune":
            features_MD = self.model_x(graph)
            return features_MD
        if self.configs.model.X_module == "DCCM_GNN":
            protein_MD_feature, residue_MD_feature = self.model_x(graph)
            return protein_MD_feature, residue_MD_feature
        if self.configs.model.X_module == "Geom2vec":
            protein_MD_feature, residue_MD_feature = self.model_x(graph[0],graph[1]) # graph = [res_rep, prot_vec]
            return protein_MD_feature, residue_MD_feature
        if self.configs.model.X_module == "Geom2vec_tematt":
            protein_MD_feature, residue_MD_feature = self.model_x(graph[0],graph[1]) # graph = [res_rep, batch_ind]
            return protein_MD_feature, residue_MD_feature
        if self.configs.model.X_module == "Geom2vec_temlstmatt":
            protein_MD_feature, residue_MD_feature = self.model_x(graph[0],graph[1]) # graph = [res_rep, batch_ind]
            return protein_MD_feature, residue_MD_feature


class GVP_CLASSIFICATION(nn.Module):
    def __init__(self, configs):
        super(GVP_CLASSIFICATION, self).__init__()
        self.model_struct =  GVPEncoder(configs=configs, residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                              residue_out_dim=configs.model.residue_out_dim,
                              protein_out_dim=configs.model.protein_out_dim,
                              protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                              residue_num_projector=configs.model.residue_num_projector,
                              protein_num_projector=configs.model.protein_num_projector,
                              seqlen=configs.model.esm_encoder.max_length)
        
        dim_mlp = configs.model.struct_encoder.node_h_dim[0]
        
        #frozen all projectors of model_struct
        for name, param in self.model_struct.projectors_protein.named_parameters():
                param.requires_grad = False
        
        for name, param in self.model_struct.projectors_residue.named_parameters():
                param.requires_grad = False
        
        self.classification_layer = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_encoder.class_inner_dim, out_dim=configs.model.class_out_dim,
                                          num_layers=configs.model.class_num_projector)
    
    def forward(self, graph=None, mode='classification', return_embedding=False,esm2_representation=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            if self.model_struct.use_seq and self.model_struct.seq_embed_mode != "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                          seq=graph.seq)
            elif self.model_struct.use_seq and self.model_struct.seq_embed_mode == "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                          seq=esm2_representation
                                                          )
            else:
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                          seq=None)
            
            # graph_feature=scatter_mean(residue_feature, batch, dim=0)
            graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
            class_logits = self.classification_layer(graph_feature_embedding)
            return class_logits, graph_feature_embedding

class GVP_CONTRASTIVE_LEARNING(nn.Module):
    def __init__(self, configs):
        super(GVP_CONTRASTIVE_LEARNING, self).__init__()
        self.model_struct = GVPEncoder(configs=configs,
                                       residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                                       residue_out_dim=configs.model.residue_out_dim,
                                       protein_out_dim=configs.model.protein_out_dim,
                                       protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                                       residue_num_projector=configs.model.residue_num_projector,
                                       protein_num_projector=configs.model.protein_num_projector,
                                       seqlen=configs.model.esm_encoder.max_length)
        
        if configs.model.struct_encoder.use_seq.enable is True and \
           configs.model.struct_encoder.use_seq.seq_embed_mode == "ESM2":
           self.esm2model,self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
           self.batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=configs.model.esm_encoder.max_length)
        
        dim_mlp = configs.model.struct_encoder.node_h_dim[0]
        self.max_length = configs.model.esm_encoder.max_length
        # frozen all projectors of model_struct
        for name, param in self.model_struct.projectors_protein.named_parameters():
            param.requires_grad = False

        for name, param in self.model_struct.projectors_residue.named_parameters():
            param.requires_grad = False

        #self.classification_layer = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_encoder.class_inner_dim,
        #                                    out_dim=configs.model.class_out_dim,
        #                                    num_layers=configs.model.class_num_projector)

        # Yichuan: projection head for contrastive learning
        #self.projection_head = LayerNormNet(configs)
        self.batch_size = configs.train_settings.batch_size
        self.n_pos = configs.model.contrastive_learning.n_pos
        self.n_neg = configs.model.contrastive_learning.n_neg
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=dim_mlp)

        # Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_mlp, nhead=configs.model.struct_encoder.transformer_num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.model.struct_encoder.transformer_num_layers)

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding
    
    @staticmethod
    def separate_features(batched_features, batch):
        # Get the number of nodes in each graph
        node_counts = batch.bincount().tolist()
        # Split the features tensor into separate tensors for each graph
        features_list = torch.split(batched_features, node_counts)
        return features_list
    
    def merge_features(self, features_list):
        # Pad tensors
        padded_tensors = []
        for t in features_list:
            #print("seqlength="+str(t.size(0)))
            if t.size(0) < self.max_length:
                size_diff = self.max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1), device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            else:
                t_padded = t
            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation
        
        # Concatenate tensors
        result = torch.cat(padded_tensors, dim=0)
        return result
    def reorganize_batch(self, graph_feature_embedding,n_pos=None,n_neg=None):
        """
        [bsz * n_all, out_dim] -> [bsz, n_all, out_dim]
        """
        # batch_size = self.batch_size
        if n_pos is not None and n_neg is not None:
            n_all = 1
        else:
            n_all = 1 + self.n_pos + self.n_neg

        real_batch_size = graph_feature_embedding.size(0) // n_all

        # if graph_feature_embedding.size(0) != batch_size * n_all:
        #     raise ValueError(f"Input tensor size {graph_feature_embedding.size(0)} does not match expected dimensions {batch_size * n_all} (batch_size={batch_size}, n_all={n_all}).")

        if graph_feature_embedding.size(0) % n_all != 0:
            raise ValueError(f"Input tensor size {graph_feature_embedding.size(0)} is not divisible by {n_all}. There might be an unexpected error in the data preparation pipeline.")

        reorganized_list = []

        for i in range(real_batch_size):
            start_idx = i * n_all
            end_idx = start_idx + n_all
            reorganized_list.append(graph_feature_embedding[start_idx:end_idx])

        reorganized_graph_feature_embedding = torch.stack(reorganized_list)

        # raise ValueError(reorganized_graph_feature_embedding.shape)

        return reorganized_graph_feature_embedding

    def forward(self, graph=None, mode='classification', return_embedding=False, batch_tokens=None,n_pos=None,n_neg=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            if self.model_struct.use_seq and self.model_struct.seq_embed_mode != "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=graph.seq)
            elif self.model_struct.use_seq and self.model_struct.seq_embed_mode == "ESM2":
                #to do add esm2_representation
                with torch.no_grad():
                    esm2_representation = self.esm2model(batch_tokens,repr_layers=[self.esm2model.num_layers], return_contacts=False)\
                                          ['representations'][self.esm2model.num_layers]
                    mask = ((batch_tokens != self.alphabet.padding_idx) & (batch_tokens != self.alphabet.cls_idx) & (
                              batch_tokens != self.alphabet.eos_idx))  # use this in v2 training
                    
                    #print(esm2_representation)
                    #print(esm2_representation.shape)
                    esm2_representation = esm2_representation[mask]
                
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=esm2_representation #should be`torch.Tensor` of shape [num_nodes]
                                                                       )
            else:
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                                       seq=None)
            
            #graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
            #projection_head = self.projection_head(self.reorganize_batch(graph_feature_embedding))
            #add transformer layer need to move inside gvp model later
            x = self.separate_features(residue_feature_embedding, graph.batch)
            #print("seperate_features")
            x= self.merge_features(x)
            #print("merge_features")
            #print(x.shape)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            graph_feature_embedding = torch.mean(x, dim=1) #return  [bsz * n_all, out_dim] 
            projection_head = self.reorganize_batch(graph_feature_embedding,n_pos,n_neg)
            # raise ValueError(projection_head.shape)

            return projection_head

#to test the clean contrastive learning strategy.
#use this to test clean and inference_mode
#to do if GVP+ESM2 not good test this
class ESM2_CONTRASTIVE_LEARNING(nn.Module):
    def __init__(self, configs):
        super(ESM2_CONTRASTIVE_LEARNING, self).__init__()
        self.esm2model = esm.pretrained.esm2_t33_650M_UR50D()
        dim_mlp = 1280
        self.max_length = configs.model.esm_encoder.max_length
        # frozen all projectors of model_struct
        for name, param in self.esm2model.named_parameters():
            param.requires_grad = False

        self.classification_layer = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_encoder.class_inner_dim,
                                            out_dim=configs.model.class_out_dim,
                                            num_layers=configs.model.class_num_projector)

        # Yichuan: projection head for contrastive learning
        #self.projection_head = LayerNormNet(configs)
        self.batch_size = configs.train_settings.batch_size
        self.n_pos = configs.model.contrastive_learning.n_pos
        self.n_neg = configs.model.contrastive_learning.n_neg
        # Positional encoding
        #self.positional_encoding = PositionalEncoding(d_model=dim_mlp)

        # Transformer Encoder layer
        #encoder_layer = nn.TransformerEncoderLayer(d_model=dim_mlp, nhead=configs.model.struct_encoder.transformer_num_heads)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.model.struct_encoder.transformer_num_layers)

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding
    
    @staticmethod
    def separate_features(batched_features, batch):
        # Get the number of nodes in each graph
        node_counts = batch.bincount().tolist()
        # Split the features tensor into separate tensors for each graph
        features_list = torch.split(batched_features, node_counts)
        return features_list
    
    def merge_features(self, features_list):
        # Pad tensors
        padded_tensors = []
        for t in features_list:
            #print("seqlength="+str(t.size(0)))
            if t.size(0) < self.max_length:
                size_diff = self.max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1), device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            else:
                t_padded = t
            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation
        
        # Concatenate tensors
        result = torch.cat(padded_tensors, dim=0)
        return result
    def reorganize_batch(self, graph_feature_embedding,n_pos=None,n_neg=None):
        """
        [bsz * n_all, out_dim] -> [bsz, n_all, out_dim]
        """
        # batch_size = self.batch_size
        if n_pos is not None and n_neg is not None:
            n_all = 1
        else:
            n_all = 1 + self.n_pos + self.n_neg

        real_batch_size = graph_feature_embedding.size(0) // n_all

        # if graph_feature_embedding.size(0) != batch_size * n_all:
        #     raise ValueError(f"Input tensor size {graph_feature_embedding.size(0)} does not match expected dimensions {batch_size * n_all} (batch_size={batch_size}, n_all={n_all}).")

        if graph_feature_embedding.size(0) % n_all != 0:
            raise ValueError(f"Input tensor size {graph_feature_embedding.size(0)} is not divisible by {n_all}. There might be an unexpected error in the data preparation pipeline.")

        reorganized_list = []

        for i in range(real_batch_size):
            start_idx = i * n_all
            end_idx = start_idx + n_all
            reorganized_list.append(graph_feature_embedding[start_idx:end_idx])

        reorganized_graph_feature_embedding = torch.stack(reorganized_list)

        # raise ValueError(reorganized_graph_feature_embedding.shape)

        return reorganized_graph_feature_embedding

    def forward(self, graph=None, mode='classification', return_embedding=False, batch_tokens=None,n_pos=None,n_neg=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            with torch.no_grad():
                esm2_representation = self.esm2model(batch_tokens,repr_layers=[self.esm2model.num_layers], return_contacts=False)
                mask = ((batch_tokens != self.esm2model.alphabet.padding_idx) & (batch_tokens != self.esm2model.alphabet.cls_idx) & (
                          batch_tokens != self.esm2model.alphabet.eos_idx))  # use this in v2 training
                esm2_representation = esm2_representation[mask]
            
            #graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
            #projection_head = self.projection_head(self.reorganize_batch(graph_feature_embedding))
            #add transformer layer need to move inside gvp model later
            x = self.separate_features(residue_feature_embedding, graph.batch)
            #print("seperate_features")
            x= self.merge_features(x)
            #print("merge_features")
            #print(x.shape)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            graph_feature_embedding = torch.mean(x, dim=1) #return  [bsz * n_all, out_dim] 
            projection_head = self.reorganize_batch(graph_feature_embedding,n_pos,n_neg)
            # raise ValueError(projection_head.shape)

            return projection_head

class GVP_PRETRAIN(nn.Module):
    def __init__(self, configs):
        super(GVP_PRETRAIN, self).__init__()
        self.model_struct =  GVPEncoder(configs=configs, residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                              residue_out_dim=configs.model.residue_out_dim,
                              protein_out_dim=configs.model.protein_out_dim,
                              protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                              residue_num_projector=configs.model.residue_num_projector,
                              protein_num_projector=configs.model.protein_num_projector,
                              seqlen=configs.model.esm_encoder.max_length)
        
        self.max_length = configs.model.esm_encoder.max_length
        self.decode_mode = configs.model.struct_decoder.mode
        dim_mlp = configs.model.struct_encoder.node_h_dim[0]
        self.transformer_after_decoder = configs.model.struct_decoder.transformer_after_decoder
        if configs.model.struct_decoder.transformer_num_layers>0:
            self.transformer_enable = True
        else:
            self.transformer_enable = False
        #frozen all projectors of model_struct
        for name, param in self.model_struct.projectors_protein.named_parameters():
                param.requires_grad = False
        
        for name, param in self.model_struct.projectors_residue.named_parameters():
                param.requires_grad = False
        
        if self.transformer_after_decoder and self.decode_mode =="struct":
            trasnsformer_input_dim = 12 #should not be good for number of head = 4 but can still try!
        else:
            trasnsformer_input_dim = dim_mlp
        
        # Transformer Decoder
        if self.transformer_enable: #do we need trasnsformer decoder??
            self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, trasnsformer_input_dim) * .02)
            self.decoder_layer = nn.TransformerEncoderLayer(
              d_model=trasnsformer_input_dim,
              nhead=configs.model.struct_decoder.transformer_num_heads,
              dim_feedforward=configs.model.struct_decoder.transformer_dim_feedforward,
              activation=configs.model.struct_decoder.transformer_actfun
              )
            self.transformer_layer = nn.TransformerEncoder(self.decoder_layer, num_layers=configs.model.struct_decoder.transformer_num_layers)
        
        if self.decode_mode == "struct" or self.decode_mode =="struct_gvp": #decode as 3D structure first
            self.struct_decoder = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_decoder.inner_dim, out_dim=3*4,num_layers=configs.model.struct_decoder.num_layers)
        elif self.decode_mode =="gvp":
            self.struct_decoder = MoBYMLP(in_dim=dim_mlp, inner_dim=configs.model.struct_decoder.inner_dim, out_dim=dim_mlp, num_layers=configs.model.struct_decoder.num_layers)
    
    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding
    
    @staticmethod
    def separate_features(batched_features, batch):
        # Get the number of nodes in each graph
        node_counts = batch.bincount().tolist()
        # Split the features tensor into separate tensors for each graph
        features_list = torch.split(batched_features, node_counts)
        return features_list
    
    def merge_features(self, features_list):
        # Pad tensors
        padded_tensors = []
        for t in features_list:
            if t.size(0) < self.max_length:
                size_diff = self.max_length - t.size(0)
                pad = torch.zeros(size_diff, t.size(1), device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            else:
                t_padded = t
            padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation
        
        # Concatenate tensors
        result = torch.cat(padded_tensors, dim=0)
        return result
    
    def forward(self, graph=None, mode='classification', return_embedding=False,esm2_representation=None):
        """
        graph: torch_geometric batchdasta
        mode can be structure or classification
        """
        if mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        elif mode == 'classification':
            nodes = (graph.node_s, graph.node_v)
            edges = (graph.edge_s, graph.edge_v)
            if self.model_struct.use_seq and self.model_struct.seq_embed_mode != "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                          seq=graph.seq)
            elif self.model_struct.use_seq and self.model_struct.seq_embed_mode == "ESM2":
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                          seq=esm2_representation
                                                          )
            else:
                residue_feature_embedding = self.model_struct.backbone(nodes, graph.edge_index, edges,
                                                          seq=None)
            
            if self.transformer_after_decoder:
                  x = self.struct_decoder(residue_feature_embedding)
                  x = self.separate_features(x, graph.batch)
                  x= self.merge_features(x)
                  if self.transformer_enable:
                     x = self.drop_positional_encoding(x,self.pos_embed_decoder)
                     x = self.transformer_layer(x)
            else:
                  x = self.separate_features(residue_feature_embedding, graph.batch)
                  x= self.merge_features(x)
                  #print("merge shape") #[64,512,100]
                  #print(x.shape)
                  if self.transformer_enable:
                     x = self.drop_positional_encoding(x,self.pos_embed_decoder)
                     x = self.transformer_layer(x)
                  
                  #print("x shape") #[64,512,100]
                  residue_dim = x.shape #B,L,H
                  x = self.struct_decoder(x.view(-1, residue_dim[-1])).view(residue_dim[0],residue_dim[1],-1)
                  #print(x.shape) #torch.Size([128, 512, 100])
            
            if self.decode_mode == "struct_gvp":
                x = x #to do
            
            return x


def info_nce_loss(features_struct, features_seq, n_views, temperature, accelerator):  # loss function
    batch_size = len(features_struct)

    labels = torch.cat([torch.arange(batch_size, device=accelerator.device) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(accelerator.device)

    features_struct = F.normalize(features_struct, dim=1)
    features_seq = F.normalize(features_seq, dim=1)
    features = torch.cat([features_struct, features_seq], dim=0)

    similarity_matrix = torch.matmul(features, features.T)

    # Discard the main diagonal from both labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=accelerator.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # Select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=accelerator.device)

    logits = logits / temperature
    return logits, labels

def clip_infonce(features_struct, features_seq, temperature, accelerator):
    # features_struct, features_seq: [B, D]
    B = features_struct.shape[0]

    z_s = F.normalize(features_struct, dim=1)
    z_q = F.normalize(features_seq, dim=1)

    # cross-modal similarity only: [B, B]
    logits = (z_s @ z_q.T) / temperature

    targets = torch.arange(B, device=accelerator.device)

    # loss_s2q = F.cross_entropy(logits, targets)
    # loss_q2s = F.cross_entropy(logits.T, targets)

    # loss = 0.5 * (loss_s2q + loss_q2s)
    return logits, targets
    # return loss

class LayerNormNet(nn.Module):
    """
    YICHUAN NEW
    From https://github.com/tttianhao/CLEAN
    """

    def __init__(self, configs):
        super(LayerNormNet, self).__init__()
        self.out_dim = configs.model.contrastive_learning.out_dim
        self.hidden_dim = configs.model.contrastive_learning.hidden_dim
        self.drop_out = configs.model.contrastive_learning.drop_out
        self.device = 'cuda'
        self.dtype = torch.float32
        feature_dim = configs.model.struct_encoder.node_h_dim[0]
        self.fc1 = nn.Linear(feature_dim, self.hidden_dim, dtype=self.dtype, device=self.device)
        self.ln1 = nn.LayerNorm(self.hidden_dim, dtype=self.dtype, device=self.device)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim,
                             dtype=self.dtype, device=self.device)
        self.ln2 = nn.LayerNorm(self.hidden_dim, dtype=self.dtype, device=self.device)
        self.fc3 = nn.Linear(self.hidden_dim, self.out_dim, dtype=self.dtype, device=self.device)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_gvpclassification_model(logging,configs,accelerator):
    gvp_classification = GVP_CLASSIFICATION(configs=configs)
    
    if accelerator.is_main_process:
        print_trainable_parameters(gvp_classification, logging) #7,443,191
        print_trainable_parameters(gvp_classification.model_struct, logging) #616,794
        print_trainable_parameters(gvp_classification.classification_layer,logging) #6,826,397
        print_trainable_parameters(gvp_classification.model_struct.backbone, logging) #616,794
    
    return gvp_classification  # model_seq, model_struct #, new_model

def prepare_gvppretrain_model(logging,configs,accelerator):
    gvp_pretrain = GVP_PRETRAIN(configs=configs)
    
    if accelerator.is_main_process:
        print_trainable_parameters(gvp_pretrain, logging) #2,822,154
        print_trainable_parameters(gvp_pretrain.model_struct, logging) #616,794
    
    return gvp_pretrain  # model_seq, model_struct #, new_model



def prepare_models(logging, configs, accelerator):
    # Use ESM2 for sequence
    model_seq = ESM2(configs.model.esm_encoder.model_name,
                     accelerator=accelerator,
                     residue_inner_dim=configs.model.esm_encoder.residue_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs, logging=logging)

    if configs.model.X_module == "MD" or configs.model.X_module == "vivit":
        model_MD = VIVIT(configs.model.MD_encoder.model_name,
                     accelerator=accelerator,
                     residue_inner_dim=configs.model.MD_encoder.residue_inner_dim,
                     protein_inner_dim=configs.model.MD_encoder.protein_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs, logging=logging,
                     dim_mlp=configs.model.MD_encoder.dim_mlp)
        
        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_MD, logging)
        
        simclr = SimCLR(model_seq, model_MD, configs=configs)
    elif configs.model.X_module == "MD_tune":
        model_MD = VIVIT_tune(configs.model.MD_encoder.model_name,
                     accelerator=accelerator,
                     residue_inner_dim=configs.model.MD_encoder.residue_inner_dim,
                     protein_inner_dim=configs.model.MD_encoder.protein_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs, logging=logging,
                     dim_mlp=configs.model.MD_encoder.dim_mlp)
        
        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_MD, logging)
        
        simclr = SimCLR(model_seq, model_MD, configs=configs)
    elif configs.model.X_module == "VJEPA2_tune":
        model_MD = VJEPA2_tune(
                     protein_inner_dim=configs.model.MD_encoder.protein_inner_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs,
                     dim_mlp=configs.model.MD_encoder.dim_mlp)
        
        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_MD, logging)
        
        simclr = SimCLR(model_seq, model_MD, configs=configs)
    elif configs.model.X_module == "DCCM_GNN":
        model_DCCM_GNN = DCCM_GNN(configs=configs, node_input_dim=configs.model.DCCM_GNN_encoder.node_input_dim,
                                  edge_dim = configs.model.DCCM_GNN_encoder.edge_dim,
                                  hidden_dim = configs.model.DCCM_GNN_encoder.hidden_dim,
                                  output_dim = configs.model.DCCM_GNN_encoder.output_dim,
                                  num_layers = configs.model.DCCM_GNN_encoder.num_layers,
                                  residue_inner_dim = configs.model.DCCM_GNN_encoder.residue_inner_dim,
                                  residue_out_dim=configs.model.residue_out_dim,
                                  residue_num_projector=configs.model.residue_num_projector,
                                  protein_inner_dim=configs.model.DCCM_GNN_encoder.protein_inner_dim,
                                  protein_out_dim=configs.model.protein_out_dim,
                                  protein_num_projector=configs.model.protein_num_projector,
                                  pooling = configs.model.DCCM_GNN_encoder.pooling)
        
        simclr = SimCLR(model_seq, model_DCCM_GNN, configs=configs)

        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_DCCM_GNN, logging)
    elif configs.model.X_module == 'Geom2vec':
        model_Geom2vec = Geom2vec(configs=configs,
                     residue_inner_dim=configs.model.Geom2vec_encoder.residue_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_inner_dim=configs.model.Geom2vec_encoderr.protein_inner_dim,
                     protein_num_projector=configs.model.protein_num_projector)
        
        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_MD, logging)
        
        simclr = SimCLR(model_seq, model_Geom2vec, configs=configs)
    elif configs.model.X_module == 'Geom2vec_tematt':
        model_Geom2vec_tematt = Geom2vec_tematt(configs=configs,
                                                input_dim=configs.model.Geom2vec_tematt_encoder.cnn_input_dim, 
                                                hidden_dim=configs.model.Geom2vec_tematt_encoder.cnn_hidden_dim, 
                                                output_dim=configs.model.Geom2vec_tematt_encoder.cnn_output_dim,
                                                temporal_dim=configs.model.Geom2vec_tematt_encoder.temporal_dim,
                     residue_inner_dim=configs.model.Geom2vec_tematt_encoder.residue_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_inner_dim=configs.model.Geom2vec_tematt_encoder.protein_inner_dim,
                     protein_num_projector=configs.model.protein_num_projector)
        
        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_Geom2vec_tematt, logging)
        
        simclr = SimCLR(model_seq, model_Geom2vec_tematt, configs=configs)
    elif configs.model.X_module == 'Geom2vec_temlstmatt':
        model_Geom2vec_temlstmatt = Geom2vec_temlstmatt(configs=configs,
                                                input_dim=configs.model.Geom2vec_temlstmatt_encoder.input_dim, 
                                                hidden_dim=configs.model.Geom2vec_temlstmatt_encoder.hidden_dim, 
                                                output_dim=configs.model.Geom2vec_temlstmatt_encoder.output_dim,
                     residue_inner_dim=configs.model.Geom2vec_temlstmatt_encoder.residue_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_inner_dim=configs.model.Geom2vec_temlstmatt_encoder.protein_inner_dim,
                     protein_num_projector=configs.model.protein_num_projector)
        
        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_Geom2vec_temlstmatt, logging)
        
        simclr = SimCLR(model_seq, model_Geom2vec_temlstmatt, configs=configs)
    elif configs.model.X_module == "structure":
        model_struct = GVPEncoder(configs=configs, residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                              residue_out_dim=configs.model.residue_out_dim,
                              protein_out_dim=configs.model.protein_out_dim,
                              protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                              residue_num_projector=configs.model.residue_num_projector,
                              protein_num_projector=configs.model.protein_num_projector,
                              seqlen=configs.model.esm_encoder.max_length)
        
        simclr = SimCLR(model_seq, model_struct, configs=configs)

        if accelerator.is_main_process:
          print_trainable_parameters(model_seq, logging)
          print_trainable_parameters(model_struct, logging)
    
    return simclr  # model_seq, model_struct #, new_model

def prepare_gvp_contrastive_learning_model(logging, configs, accelerator):
    gvp_contrastive_learning = GVP_CONTRASTIVE_LEARNING(configs=configs)
    if accelerator.is_main_process:
        print_trainable_parameters(gvp_contrastive_learning, logging)  # 7,443,191
        print_trainable_parameters(gvp_contrastive_learning.model_struct, logging)  # 616,794
        #print_trainable_parameters(gvp_contrastive_learning.classification_layer, logging)  # 6,826,397
        print_trainable_parameters(gvp_contrastive_learning.model_struct.backbone, logging)  # 616,794
        #trainable params:  1,064,538 || all params:  2,541,722 || trainable%: 41.882550491359794
        #trainable params:  616,794 || all params:  2,093,978 || trainable%: 29.45561032637401
        #trainable params:  616,794 || all params:  616,794 || trainable%: 100.0


    return gvp_contrastive_learning  # model_seq, model_struct #, new_model

def info_nce_weights(features_struct, features_seq, device, n_views=2, mode="positive"):  # loss function
    """
    mode = "positive" only apply plddt weights on positive pairs
    mode = "all" apply plddt weights on all pairs
    """
    # print("batch_size="+str(self.args.batch_size))
    features_struct = features_struct.to(device)
    features_seq = features_seq.to(device)
    batch_size = len(features_struct)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)],
                       dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = torch.cat([features_struct, features_seq], dim=0)
    if len(features.shape) < 2:
        features = features.unsqueeze(1)

    avg_weights_matrix = (features + features.T) / 2.0

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    avg_weights_matrix = avg_weights_matrix[~mask].view(avg_weights_matrix.shape[0], -1)
    # apply plddt weight on positives
    positives = avg_weights_matrix[labels.bool()].view(labels.shape[0], -1)
    # apply plddt weight on negatives
    # negatives = avg_weights_matrix[~labels.bool()].view(avg_weights_matrix.shape[0], -1)
    # only apply plddt weight on positives
    if mode == "positive":
        negative_weights = torch.ones(avg_weights_matrix.shape).to(device)
        negatives = negative_weights[~labels.bool()].view(negative_weights.shape[0], -1)
    elif mode == "all":
        negatives = avg_weights_matrix[~labels.bool()].view(avg_weights_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    return logits


if __name__ == '__main__':
    print('test')
