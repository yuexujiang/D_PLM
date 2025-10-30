import os

import torch.nn as nn
import torch
import esm
import esm_adapterH
import torch.backends.cudnn as cudnn
import numpy as np
# import pylab
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import math
from collections import OrderedDict
from inspect import signature
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from Bio.PDB import PDBParser, PPBuilder


def get_embedding(embed_dim, num_embeddings):
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embed_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

    return emb


def patch_num_32(n):
    remainder = n % 32
    if remainder == 0:
        return n // 32
    else:
        return (n + (32 - remainder)) // 32


def scatter_labeled_z(z_batch, colors, filename="test_plot"):
    fig = plt.gcf()
    plt.switch_backend('Agg')
    fig.set_size_inches(3.5, 3.5)
    plt.clf()
    for n in range(z_batch.shape[0]):
        result = plt.scatter(z_batch[n, 0], z_batch[n, 1], color=colors[n], s=50, marker="o", edgecolors='none')

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig(filename)
    plt.close()
    # pylab.show()


class ESM2_representation_lora(nn.Module):  # for sequence
    def __init__(self, checkpoint_path=None, esm2_pretrain=None, esm2_pretrain_local=None, existingmodel=None,
                 pool_mode=2, device='cpu', max_length=512):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(ESM2_representation_lora, self).__init__()
        if existingmodel is None:  # need to load the current model from checkpoint
            esm2_dict = {"esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D(),  # 33 layers embedding=1280
                         "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D(),  # 30 layers embedding=640
                         "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D(),  # 12 layers embedding=480
                         "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D(),  # 6 layers embedding = 320
                         }

            if esm2_pretrain_local is None:
                esm2, self.alphabet = esm2_dict[esm2_pretrain]  # alphabet.all_toks
            else:
                # print("load esm2 model from local dir")
                esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)

            self.peft_model = PeftModel.from_pretrained(esm2, checkpoint_path)
            self.peft_model.eval()
        else:
            self.peft_model = existingmodel.peft_model
            self.alphabet = self.peft_model.alphabet

        self.peft_model.eval()
        self.num_layers = self.peft_model.num_layers
        self.pool_mode = pool_mode
        if self.pool_mode == 3:
            self.pe = get_embedding(self.esm2.embed_dim, max_length + 2).to(device)

    def forward(self, x):
        residue_feature = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)['representations'][
            self.num_layers]
        if self.pool_mode == 1:
            graph_feature = residue_feature[:, 0, :]
        elif self.pool_mode == 2:
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom  # remove padding
        elif self.pool_mode == 3:
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature = torch.sum(
                residue_feature * self.pe[:residue_feature.shape[1], :].unsqueeze(0) * mask.unsqueeze(-1),
                dim=1) / denom  # remove padding and normlize

        return graph_feature


# model_prepresent=ESM2_representation_lora(checkpoint_path,esm2_pretrain,esm2_pretrain_local)

class ESM2_representation_finetune(nn.Module):  # for sequence
    def __init__(self, esm2_pretrain=None, esm2_pretrain_local=None, existingmodel=None, pool_mode=2, device='cpu',
                 max_length=512):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(ESM2_representation_finetune, self).__init__()
        if existingmodel is None:
            esm2_dict = {"esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D(),  # 33 layers embedding=1280
                         "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D(),  # 30 layers embedding=640
                         "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D(),  # 12 layers embedding=480
                         "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D(),  # 6 layers embedding = 320
                         }

            if esm2_pretrain_local is None:
                self.esm2, self.alphabet = esm2_dict[esm2_pretrain]  # alphabet.all_toks
            else:
                # print("load esm2 model from local dir")
                self.esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)

        else:
            self.esm2 = existingmodel.esm2
            self.alphabet = self.esm2.alphabet

        self.num_layers = self.esm2.num_layers
        self.pool_mode = pool_mode
        if self.pool_mode == 3:
            self.pe = get_embedding(self.esm2.embed_dim, max_length + 2).to(device)

        self.esm2.eval()

    def forward(self, x):
        residue_feature = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)['representations'][
            self.num_layers]
        if self.pool_mode == 1:
            graph_feature = residue_feature[:, 0, :]
        elif self.pool_mode == 2:
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom  # remove padding
        elif self.pool_mode == 3:
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature = torch.sum(
                residue_feature * self.pe[:residue_feature.shape[1], :].unsqueeze(0) * mask.unsqueeze(-1),
                dim=1) / denom  # remove padding and normlize

        return graph_feature


class ESM2_representation_adapterH(nn.Module):  # for sequence
    def __init__(self, num_end_adapter_layers=33, esm2_pretrain=None, esm2_pretrain_local=None, existingmodel=None,
                 pool_mode=2, device='cpu', max_length=512):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(ESM2_representation_adapterH, self).__init__()
        if existingmodel is None:
            esm2_dict = {"esm2_t33_650M_UR50D": esm_adapterH.pretrained.esm2_t33_650M_UR50D(num_end_adapter_layers),
                         # 33 layers embedding=1280
                         "esm2_t30_150M_UR50D": esm_adapterH.pretrained.esm2_t30_150M_UR50D(num_end_adapter_layers),
                         # 30 layers embedding=640
                         "esm2_t12_35M_UR50D": esm_adapterH.pretrained.esm2_t12_35M_UR50D(num_end_adapter_layers),
                         # 12 layers embedding=480
                         "esm2_t6_8M_UR50D": esm_adapterH.pretrained.esm2_t6_8M_UR50D(num_end_adapter_layers),
                         # 6 layers embedding = 320
                         }

            if esm2_pretrain_local is None:
                self.esm2, self.alphabet = esm2_dict[esm2_pretrain]  # alphabet.all_toks
            else:
                # print("load esm2 model from local dir")
                self.esm2, self.alphabet = esm_adapterH.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)

        else:
            self.esm2 = existingmodel.esm2
            self.alphabet = self.esm2.alphabet

        self.num_layers = self.esm2.num_layers
        self.pool_mode = pool_mode
        if self.pool_mode == 3:
            # print("pool_mode is 3")
            self.pe = get_embedding(self.esm2.embed_dim, max_length + 2).to(device)

        self.esm2.eval()

    def forward(self, x):
        residue_feature = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)['representations'][
            self.num_layers]
        if self.pool_mode == 1:
            graph_feature = residue_feature[:, 0, :]
        elif self.pool_mode == 2:
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom  # remove padding
        elif self.pool_mode == 3:
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature = torch.sum(
                residue_feature * self.pe[:residue_feature.shape[1], :].unsqueeze(0) * mask.unsqueeze(-1),
                dim=1) / denom  # remove padding and normlize

        return graph_feature


# model_prepresent=ESM2_representation_finetune(esm2_pretrain,esm2_pretrain_local)

def init_model(esmType, checkpoint_path=None,
               esm2_pretrain=None,
               esm2_pretrain_local=None,
               existingmodel=None,
               num_end_adapter_layers=33,
               maxlength=500,
               pool_mode=2,
               device='cpu'):
    if esmType == "lora":
        # print("modeltype=lora")
        model_prepresent = ESM2_representation_lora(checkpoint_path, esm2_pretrain, esm2_pretrain_local, existingmodel,
                                                    pool_mode=pool_mode, device=device, max_length=maxlength)
    elif esmType == "finetune":
        model_prepresent = ESM2_representation_finetune(esm2_pretrain, esm2_pretrain_local, existingmodel,
                                                        pool_mode=pool_mode, device=device, max_length=maxlength)
        if existingmodel is None and checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model_prepresent.load_state_dict(checkpoint['state_dict1'], strict=False)
    elif esmType == "adapterH":
        model_prepresent = ESM2_representation_adapterH(num_end_adapter_layers, esm2_pretrain, esm2_pretrain_local,
                                                        existingmodel, pool_mode=pool_mode, device=device,
                                                        max_length=maxlength)
        if existingmodel is None and checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            if np.sum(["adapter_layer_dict" in key for key in checkpoint[
                'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
                new_ordered_dict = OrderedDict()
                for key, value in checkpoint['state_dict1'].items():
                    if "adapter_layer_dict" not in key:
                        new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                        new_ordered_dict[new_key] = value
                    else:
                        new_ordered_dict[key] = value

                model_prepresent.load_state_dict(new_ordered_dict, strict=False)
            else:
                model_prepresent.load_state_dict(checkpoint['state_dict1'], strict=False)

            print("checkpoints were loaded from " + checkpoint_path)

    model_prepresent.eval()
    alphabet = model_prepresent.alphabet
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=maxlength)

    return model_prepresent, batch_converter

def evaluate_with_cath_more_md(out_figure_path, steps, accelerator, batch_size, model, batch_converter, cathpath):
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    seq_embeddings = []
    input = open(cathpath)
    labels = []
    seqs = []
    for line in input:
        labels.append(line[1:].split("|")[0].strip())
        line = next(input)
        seqs.append(line.strip())
    
    from torch.utils.data import DataLoader, Dataset
    
    class DataSet(Dataset):
        def __init__(self, pad_seq):
            self.pad_seq = pad_seq
        
        def __len__(self):
            return len(self.pad_seq)
        
        def __getitem__(self, index):
            return index, self.pad_seq[index]
    
    dataset_val = DataSet(seqs)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    # val_loader = accelerator.prepare(val_loader)
    
    # model_prepresent.eval()
    for batch in val_loader:
        batch_seq = [(batch[0][i].item(), str(batch[1][i])) for i in range(len(batch[0]))]
        with torch.inference_mode():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_tokens = batch_tokens.to(accelerator.device)
            model_signature = signature(model.forward)
            if 'batch_tokens' in model_signature.parameters:
                _, _, features_seq, _  = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
            else:
                _, _, features_seq, _  = model(batch_tokens, mode='sequence', return_embedding=True)
            
            seq_embeddings.extend(features_seq.cpu().detach().numpy())
    
    seq_embeddings = np.asarray(seq_embeddings)
    mdel = TSNE(n_components=2, random_state=0, init='random',method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []
    tol_archi_seq = {"3.30", "3.40", "1.10", "3.10", "2.60"}
    tol_fold_seq = {"1.10.10", "3.30.70", "2.60.40", "2.60.120", "3.40.50"}
    
    for digit_num in [1, 2, 3]:  # first number of digits
        color = []
        colorid = []
        keys = {}
        colorindex = 0
        if digit_num == 1:
            ct = ["blue", "red", "black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink",
                  "cyan", "peru", "darkgray", "slategray", "gold"]
        else:
            ct = ["black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink", "cyan", "peru",
                  "darkgray", "slategray", "gold"]
        
        select_label = []
        select_index = []
        color_dict = {}
        for label in labels:
            key = ".".join([x for x in label.split(".")[0:digit_num]])
            if digit_num == 2:
                if key not in tol_archi_seq:
                    continue
            if digit_num == 3:
                if key not in tol_fold_seq:
                    continue
            
            if key not in keys:
                keys[key] = colorindex
                colorindex += 1
            
            select_label.append(keys[key])
            color.append(ct[(keys[key]) % len(ct)])
            colorid.append(keys[key])
            select_index.append(labels.index(label))
            color_dict[keys[key]] = ct[keys[key]]
        
        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))
        scatter_labeled_z(z_tsne_seq[select_index], color,
                          filename=os.path.join(out_figure_path, f"step_{steps}_CATH_{digit_num}.png"))
        # add kmeans
        kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        predicted_colors = [color_dict[label] for label in predicted_labels]
        #scatter_labeled_z(z_tsne_seq[select_index], predicted_colors,
        #                  filename=os.path.join(out_figure_path, f"step_{steps}_CATH_{digit_num}_kmpred.png"))
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)
        scores.append(silhouette_score(seq_embeddings[select_index], color))
    return scores  # [digit_num1_full,digit_num_2d,digit_num2_full,digit_num2_2d]


def evaluate_with_deaminase_md(out_figure_path, model, steps, accelerator, batch_size, batch_converter, deaminasepath):
    """
    esmType: "lora","adapter","fine-tune"
    """
    # need to install openpyxl
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    
    seq_embeddings = []
    data_frame = pd.read_excel(deaminasepath, engine='openpyxl')
    column_condition = data_frame['Used in Structure Tree'] == '√'
    labels = data_frame.loc[column_condition, 'Family '].tolist()
    seqs = data_frame.loc[column_condition, 'Domain seq in HMMSCAN'].tolist()
    
    from torch.utils.data import DataLoader, Dataset
    
    class DataSet(Dataset):
        def __init__(self, pad_seq):
            self.pad_seq = pad_seq
        
        def __len__(self):
            return len(self.pad_seq)
        
        def __getitem__(self, index):
            return index, self.pad_seq[index]
    
    dataset_val = DataSet(seqs)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    # val_loader = accelerator.prepare(val_loader)
    
    # model_prepresent.eval()
    for batch in val_loader:
        batch_seq = [(batch[0][i].item(), str(batch[1][i])) for i in range(len(batch[0]))]
        with torch.inference_mode():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_tokens = batch_tokens.to(accelerator.device)
            model_signature = signature(model.forward)
            if 'batch_tokens' in model_signature.parameters:
                _, _, features_seq, _  = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
            else:
                _, _, features_seq, _  = model(batch_tokens, mode='sequence', return_embedding=True)
            
            seq_embeddings.extend(features_seq.cpu().detach().numpy())
    
    seq_embeddings = np.asarray(seq_embeddings)
    mdel = TSNE(n_components=2, random_state=0, init='random',method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []
    color = []
    colorid = []
    keys = {}
    colorindex = 0
    
    ct = {"JAB": (154 / 255, 99 / 255, 36 / 255),
          "A_deamin": (253 / 255, 203 / 255, 110 / 255),
          "AICARFT_IMPCHas": (58 / 255, 179 / 255, 73 / 255),
          "APOBEC": (255 / 255, 114 / 255, 114 / 255),
          "Bd3614-deam": (63 / 255, 96 / 255, 215 / 255),
          "Bd3615-deam": (63 / 255, 96 / 255, 215 / 255),  # no this in the paper
          "dCMP_cyt_deam_1": (151 / 255, 130 / 255, 255 / 255),
          "dCMP_cyt_deam_2": (151 / 255, 130 / 255, 255 / 255),  # note!
          "FdhD-NarQ": (141 / 255, 23 / 255, 178 / 255),
          "Inv-AAD": (62 / 255, 211 / 255, 244 / 255),
          "LmjF365940-deam": (190 / 255, 239 / 255, 65 / 255),
          "LpxI_C": (250 / 255, 190 / 255, 212 / 255),
          "MafB19-deam": (220 / 255, 189 / 255, 255 / 255),
          "Pput2613-deam": (168 / 255, 168 / 255, 168 / 255),  # DddA-like renamed into SCP1.201_deam
          "DddA-like": (65 / 255, 150 / 255, 168 / 255),
          "TM1506": (255 / 255, 255 / 255, 0 / 255),
          "Toxin-deaminase": (126 / 255, 0, 0),
          "XOO_2897-deam": (245 / 255, 128 / 255, 46 / 255),
          "YwqJ-deaminase": (124 / 255, 124 / 255, 0)
          }
    
    color_dict = {}
    for label in labels:
        key = label
        if key not in keys:
            keys[key] = colorindex
            colorindex += 1
        
        colorid.append(keys[key])
        color.append(ct[key])
        color_dict[keys[key]] = ct[key]
    
    scores.append(calinski_harabasz_score(seq_embeddings, colorid))
    scores.append(calinski_harabasz_score(z_tsne_seq, colorid))
    scatter_labeled_z(z_tsne_seq, color, filename=os.path.join(out_figure_path, f"step_{steps}_deaminase.png"))
    # add kmeans evaluation
    kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
    # predicted_labels = kmeans.fit_predict(seq_embeddings)
    predicted_labels = kmeans.fit_predict(z_tsne_seq)
    predicted_colors = [color_dict[label] for label in predicted_labels]
    #scatter_labeled_z(z_tsne_seq, predicted_colors,
    #                  filename=os.path.join(out_figure_path, f"step_{steps}_deaminase_kmpred.png"))
    ari = adjusted_rand_score(colorid, predicted_labels)
    scores.append(ari)
    scores.append(silhouette_score(seq_embeddings, colorid))
    return scores  # [score,2d_score]


def evaluate_with_kinase_md(out_figure_path, steps, batch_converter, batch_size, kinasepath, model, accelerator):
    """
    esmType: "lora","adapter","fine-tune"
    """
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    seq_embeddings = []
    
    data_frame = pd.read_csv(kinasepath, sep="\t")
    data_frame = data_frame.drop_duplicates(subset='Kinase_Entrez_Symbol', keep='first')
    column_condition = data_frame['Kinase_domain'] != ' '
    labels = data_frame.loc[column_condition, 'Kinase_group'].tolist()
    seqs = data_frame.loc[column_condition, 'Kinase_domain'].tolist()
    
    from torch.utils.data import DataLoader, Dataset
    
    class DataSet(Dataset):
        def __init__(self, pad_seq):
            self.pad_seq = pad_seq
        
        def __len__(self):
            return len(self.pad_seq)
        
        def __getitem__(self, index):
            return index, self.pad_seq[index]
    
    dataset_val = DataSet(seqs)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    # val_loader = accelerator.prepare(val_loader)
    
    # model_prepresent.eval()
    for batch in val_loader:
        batch_seq = [(batch[0][i].item(), str(batch[1][i])) for i in range(len(batch[0]))]
        with torch.inference_mode():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_tokens = batch_tokens.to(accelerator.device)
            model_signature = signature(model.forward)
            if 'batch_tokens' in model_signature.parameters:
                _, _, features_seq, _  = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
            else:
                _, _, features_seq, _  = model(batch_tokens, mode='sequence', return_embedding=True)
            
            seq_embeddings.extend(features_seq.cpu().detach().numpy())
    
    seq_embeddings = np.asarray(seq_embeddings)
    # print(seq_embeddings.shape)
    mdel = TSNE(n_components=2, random_state=0, init='random',method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []
    
    color = []
    colorid = []
    keys = {}
    colorindex = 0
    ct = {"AGC": (154 / 255, 99 / 255, 36 / 255),
          "Atypical": (253 / 255, 203 / 255, 110 / 255),
          "Other": (58 / 255, 179 / 255, 73 / 255),
          "STE": (255 / 255, 114 / 255, 114 / 255),
          "TK": (63 / 255, 96 / 255, 215 / 255),
          "CK1": (151 / 255, 130 / 255, 255 / 255),  # note!
          "CMGC": (141 / 255, 23 / 255, 178 / 255),
          "TKL": (62 / 255, 211 / 255, 244 / 255),
          "CAMK": (190 / 255, 239 / 255, 65 / 255),
          }
    
    color_dict = {}
    for label in labels:
        key = label
        if key not in keys:
            keys[key] = colorindex
            colorindex += 1
        
        colorid.append(keys[key])
        color.append(ct[key])
        color_dict[keys[key]] = ct[key]
    
    # print(seq_embeddings.shape)
    
    scatter_labeled_z(z_tsne_seq, color, filename=os.path.join(out_figure_path, f"step_{steps}_kinase.png"))
    scores.append(calinski_harabasz_score(seq_embeddings, colorid[:len(seq_embeddings)]))
    scores.append(calinski_harabasz_score(z_tsne_seq, colorid[:len(seq_embeddings)]))
    kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
    predicted_labels = kmeans.fit_predict(z_tsne_seq)
    # predicted_labels = kmeans.fit_predict(seq_embeddings)
    predicted_colors = [color_dict[label] for label in predicted_labels]
    ari = adjusted_rand_score(colorid, predicted_labels)
    #scatter_labeled_z(z_tsne_seq, predicted_colors,
    #                  filename=os.path.join(out_figure_path, f"step_{steps}_kinase_kmpred.png"))
    scores.append(ari)
    scores.append(silhouette_score(seq_embeddings, colorid[:len(seq_embeddings)]))
    
    return scores  # [digit_num1_full,digit_num_2d,digit_num2_full,digit_num2_2d]

def evaluate_with_cath_more(out_figure_path, steps, accelerator, batch_size, model, batch_converter, cathpath):
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    seq_embeddings = []
    input = open(cathpath)
    labels = []
    seqs = []
    for line in input:
        labels.append(line[1:].split("|")[0].strip())
        line = next(input)
        seqs.append(line.strip())

    from torch.utils.data import DataLoader, Dataset

    class DataSet(Dataset):
        def __init__(self, pad_seq):
            self.pad_seq = pad_seq

        def __len__(self):
            return len(self.pad_seq)

        def __getitem__(self, index):
            return index, self.pad_seq[index]

    dataset_val = DataSet(seqs)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    # val_loader = accelerator.prepare(val_loader)

    # model_prepresent.eval()
    for batch in val_loader:
        batch_seq = [(batch[0][i].item(), str(batch[1][i])) for i in range(len(batch[0]))]
        with torch.inference_mode():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_tokens = batch_tokens.to(accelerator.device)
            _, _, features_seq, _ = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
            seq_embeddings.extend(features_seq.cpu().detach().numpy())

    seq_embeddings = np.asarray(seq_embeddings)
    mdel = TSNE(n_components=2, random_state=0, init='random',method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []
    tol_archi_seq = {"3.30", "3.40", "1.10", "3.10", "2.60"}
    tol_fold_seq = {"1.10.10", "3.30.70", "2.60.40", "2.60.120", "3.40.50"}

    for digit_num in [1, 2, 3]:  # first number of digits
        color = []
        colorid = []
        keys = {}
        colorindex = 0
        if digit_num == 1:
            ct = ["blue", "red", "black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink",
                  "cyan", "peru", "darkgray", "slategray", "gold"]
        else:
            ct = ["black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink", "cyan", "peru",
                  "darkgray", "slategray", "gold"]

        select_label = []
        select_index = []
        color_dict = {}
        for label in labels:
            key = ".".join([x for x in label.split(".")[0:digit_num]])
            if digit_num == 2:
                if key not in tol_archi_seq:
                    continue
            if digit_num == 3:
                if key not in tol_fold_seq:
                    continue

            if key not in keys:
                keys[key] = colorindex
                colorindex += 1

            select_label.append(keys[key])
            color.append(ct[(keys[key]) % len(ct)])
            colorid.append(keys[key])
            select_index.append(labels.index(label))
            color_dict[keys[key]] = ct[keys[key]]

        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))
        scatter_labeled_z(z_tsne_seq[select_index], color,
                          filename=os.path.join(out_figure_path, f"step_{steps}_CATH_{digit_num}.png"))
        # add kmeans
        kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        predicted_colors = [color_dict[label] for label in predicted_labels]
        scatter_labeled_z(z_tsne_seq[select_index], predicted_colors,
                          filename=os.path.join(out_figure_path, f"step_{steps}_CATH_{digit_num}_kmpred.png"))
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)
    return scores  # [digit_num1_full,digit_num_2d,digit_num2_full,digit_num2_2d]


def evaluate_with_deaminase(out_figure_path, model, steps, accelerator, batch_size, batch_converter, deaminasepath):
    """
    esmType: "lora","adapter","fine-tune"
    """
    # need to install openpyxl
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)

    seq_embeddings = []
    data_frame = pd.read_excel(deaminasepath, engine='openpyxl')
    column_condition = data_frame['Used in Structure Tree'] == '√'
    labels = data_frame.loc[column_condition, 'Family '].tolist()
    seqs = data_frame.loc[column_condition, 'Domain seq in HMMSCAN'].tolist()

    from torch.utils.data import DataLoader, Dataset

    class DataSet(Dataset):
        def __init__(self, pad_seq):
            self.pad_seq = pad_seq

        def __len__(self):
            return len(self.pad_seq)

        def __getitem__(self, index):
            return index, self.pad_seq[index]

    dataset_val = DataSet(seqs)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    # val_loader = accelerator.prepare(val_loader)

    # model_prepresent.eval()
    for batch in val_loader:
        batch_seq = [(batch[0][i].item(), str(batch[1][i])) for i in range(len(batch[0]))]
        with torch.inference_mode():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_tokens = batch_tokens.to(accelerator.device)
            _, _, features_seq, _ = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
            seq_embeddings.extend(features_seq.cpu().detach().numpy())

    seq_embeddings = np.asarray(seq_embeddings)
    mdel = TSNE(n_components=2, random_state=0, init='random',method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []
    color = []
    colorid = []
    keys = {}
    colorindex = 0

    ct = {"JAB": (154 / 255, 99 / 255, 36 / 255),
          "A_deamin": (253 / 255, 203 / 255, 110 / 255),
          "AICARFT_IMPCHas": (58 / 255, 179 / 255, 73 / 255),
          "APOBEC": (255 / 255, 114 / 255, 114 / 255),
          "Bd3614-deam": (63 / 255, 96 / 255, 215 / 255),
          "Bd3615-deam": (63 / 255, 96 / 255, 215 / 255),  # no this in the paper
          "dCMP_cyt_deam_1": (151 / 255, 130 / 255, 255 / 255),
          "dCMP_cyt_deam_2": (151 / 255, 130 / 255, 255 / 255),  # note!
          "FdhD-NarQ": (141 / 255, 23 / 255, 178 / 255),
          "Inv-AAD": (62 / 255, 211 / 255, 244 / 255),
          "LmjF365940-deam": (190 / 255, 239 / 255, 65 / 255),
          "LpxI_C": (250 / 255, 190 / 255, 212 / 255),
          "MafB19-deam": (220 / 255, 189 / 255, 255 / 255),
          "Pput2613-deam": (168 / 255, 168 / 255, 168 / 255),  # DddA-like renamed into SCP1.201_deam
          "DddA-like": (65 / 255, 150 / 255, 168 / 255),
          "TM1506": (255 / 255, 255 / 255, 0 / 255),
          "Toxin-deaminase": (126 / 255, 0, 0),
          "XOO_2897-deam": (245 / 255, 128 / 255, 46 / 255),
          "YwqJ-deaminase": (124 / 255, 124 / 255, 0)
          }

    color_dict = {}
    for label in labels:
        key = label
        if key not in keys:
            keys[key] = colorindex
            colorindex += 1

        colorid.append(keys[key])
        color.append(ct[key])
        color_dict[keys[key]] = ct[key]

    scores.append(calinski_harabasz_score(seq_embeddings, colorid))
    scores.append(calinski_harabasz_score(z_tsne_seq, colorid))
    scatter_labeled_z(z_tsne_seq, color, filename=os.path.join(out_figure_path, f"step_{steps}_deaminase.png"))
    # add kmeans evaluation
    kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
    # predicted_labels = kmeans.fit_predict(seq_embeddings)
    predicted_labels = kmeans.fit_predict(z_tsne_seq)
    predicted_colors = [color_dict[label] for label in predicted_labels]
    scatter_labeled_z(z_tsne_seq, predicted_colors,
                      filename=os.path.join(out_figure_path, f"step_{steps}_deaminase_kmpred.png"))
    ari = adjusted_rand_score(colorid, predicted_labels)
    scores.append(ari)
    return scores  # [score,2d_score]


def evaluate_with_kinase(out_figure_path, steps, batch_converter, batch_size, kinasepath, model, accelerator):
    """
    esmType: "lora","adapter","fine-tune"
    """
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    seq_embeddings = []

    data_frame = pd.read_csv(kinasepath, sep="\t")
    data_frame = data_frame.drop_duplicates(subset='Kinase_Entrez_Symbol', keep='first')
    column_condition = data_frame['Kinase_domain'] != ' '
    labels = data_frame.loc[column_condition, 'Kinase_group'].tolist()
    seqs = data_frame.loc[column_condition, 'Kinase_domain'].tolist()

    from torch.utils.data import DataLoader, Dataset

    class DataSet(Dataset):
        def __init__(self, pad_seq):
            self.pad_seq = pad_seq

        def __len__(self):
            return len(self.pad_seq)

        def __getitem__(self, index):
            return index, self.pad_seq[index]

    dataset_val = DataSet(seqs)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    # val_loader = accelerator.prepare(val_loader)

    # model_prepresent.eval()
    for batch in val_loader:
        batch_seq = [(batch[0][i].item(), str(batch[1][i])) for i in range(len(batch[0]))]
        with torch.inference_mode():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
            batch_tokens = batch_tokens.to(accelerator.device)
            _, _, features_seq, _ = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
            seq_embeddings.extend(features_seq.cpu().detach().numpy())

    seq_embeddings = np.asarray(seq_embeddings)
    # print(seq_embeddings.shape)
    mdel = TSNE(n_components=2, random_state=0, init='random',method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []

    color = []
    colorid = []
    keys = {}
    colorindex = 0
    ct = {"AGC": (154 / 255, 99 / 255, 36 / 255),
          "Atypical": (253 / 255, 203 / 255, 110 / 255),
          "Other": (58 / 255, 179 / 255, 73 / 255),
          "STE": (255 / 255, 114 / 255, 114 / 255),
          "TK": (63 / 255, 96 / 255, 215 / 255),
          "CK1": (151 / 255, 130 / 255, 255 / 255),  # note!
          "CMGC": (141 / 255, 23 / 255, 178 / 255),
          "TKL": (62 / 255, 211 / 255, 244 / 255),
          "CAMK": (190 / 255, 239 / 255, 65 / 255),
          }

    color_dict = {}
    for label in labels:
        key = label
        if key not in keys:
            keys[key] = colorindex
            colorindex += 1

        colorid.append(keys[key])
        color.append(ct[key])
        color_dict[keys[key]] = ct[key]

    # print(seq_embeddings.shape)

    scatter_labeled_z(z_tsne_seq, color, filename=os.path.join(out_figure_path, f"step_{steps}_kinase.png"))
    scores.append(calinski_harabasz_score(seq_embeddings, colorid[:len(seq_embeddings)]))
    scores.append(calinski_harabasz_score(z_tsne_seq, colorid[:len(seq_embeddings)]))
    kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
    predicted_labels = kmeans.fit_predict(z_tsne_seq)
    # predicted_labels = kmeans.fit_predict(seq_embeddings)
    predicted_colors = [color_dict[label] for label in predicted_labels]
    ari = adjusted_rand_score(colorid, predicted_labels)
    scatter_labeled_z(z_tsne_seq, predicted_colors,
                      filename=os.path.join(out_figure_path, f"step_{steps}_kinase_kmpred.png"))
    scores.append(ari)
    return scores  # [digit_num1_full,digit_num_2d,digit_num2_full,digit_num2_2d]


def test_evaluate_allcases():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    #"""esm2
    pool_mode = 2
    # output_img_name="ESM2_kmeans_original_"+str(pool_mode)
    # output_img_name = "ESM2_kmeans_2D_" + str(pool_mode)
    esmType = "finetune"
    #esmType = "adapterH"
    # checkpoint_path="./config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3/checkpoints/checkpoint_0162000.pth"
    # checkpoint_path="./config_14fixesmtable_sgdlr001mixlenpad_esmpool3/checkpoints/checkpoint_0152000.pth"
    #checkpoint_path = "./config_16fixesm_sgdlr001mixlenpad_esmpool2_MLMcontrast_testnolmheadbz20_run2/saved_checkpoints/checkpoint_0520000.pth"
    # output_img_name="config_16fixesm_sgdlr001mixlenpad_esmpool2_MLMcontrast_testnolmheadbz20_run2/kmeans_original/"
    #output_img_name = "config_16fixesm_sgdlr001mixlenpad_esmpool2_MLMcontrast_testnolmheadbz20_run2/kmeans_2D/"

    #"""
    class args():
        def __init__(self):
            self.num_end_adapter_layers = 16  # because adapterH has changed
            self.module_type = "MLP1"

    existingmodel, batch_converter = init_model(esmType,
                                                checkpoint_path,
                                                esm2_pretrain="esm2_t33_650M_UR50D",
                                                existingmodel=None, num_end_adapter_layers=args(),
                                                maxlength=512,
                                                pool_mode=pool_mode,
                                                )

    existingmodel.eval()
    # cathpath = "/mnt/pixstor/dbllab/duolin/CATH/seq_subfamily/selected_seq_subfamily_12_basedon_S40pdb.fa"
    cathpath = "/mnt/pixstor/dbllab/duolin/CATH/seq_subfamily/Rep_subfamily_basedon_S40pdb.fa"
    # """
    scores_cath = evaluate_with_cath_more(device, output_img_name, steps=1,
                                          esmType=esmType,
                                          batch_size=8,
                                          cathpath=cathpath,
                                          existingmodel=existingmodel,
                                          pool_mode=pool_mode,
                                          )
    # """
    deaminasepath = "/mnt/pixstor/dbllab/duolin/simCLR/Deaminases_clustering/Table_S1.xlsx"
    scores_deaminase = evaluate_with_deaminase(device, output_img_name, steps=1,
                                               esmType=esmType,
                                               batch_size=2,
                                               deaminasepath=deaminasepath,
                                               existingmodel=existingmodel,
                                               pool_mode=pool_mode,
                                               )

    kinasepath = "/mnt/pixstor/dbllab/duolin/simCLR/kinase_clustering/GPS5.0_homo_hasPK_with_kinasedomain.txt"
    scores_kinase = evaluate_with_kinase(device, output_img_name, steps=1,
                                         esmType=esmType,
                                         batch_size=2,
                                         kinasepath=kinasepath,
                                         existingmodel=existingmodel,
                                         pool_mode=pool_mode,
                                         )

    n_iter = 1
    print(
        f"step: {n_iter}\tdigit_num_1:{scores_cath[0]}({scores_cath[1]})\tdigit_num_2:{scores_cath[3]}({scores_cath[4]}\tdigit_num_3:{scores_cath[6]}({scores_cath[7]})")
    print(
        f"step:{n_iter}\tdigit_num_1_ARI:{scores_cath[2]}\tdigit_num_2_ARI:{scores_cath[5]}\tdigit_num_3_ARI:{scores_cath[8]}")
    print(f"step: {n_iter}\tdeaminase_score:{scores_deaminase[0]}({scores_deaminase[1]}) ARI:{scores_deaminase[2]}")
    print(f"step: {n_iter}\tkinase_score:{scores_kinase[0]}({scores_kinase[1]}) ARI:{scores_kinase[2]}")


def compute_repdiff(row, sequence, model, alphabet, offset_idx,wt_representation,mode="whole",similarity="correlation"):
    if ":" not in row:
       base = row[1:-1]
       if len(base)==0:
        return np.nan
       
       wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
       if idx>=len(sequence):
           return np.nan
       
       assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
       # modify the sequence
       sequence = sequence[:idx] + mt + sequence[(idx + 1) :]
    else:
        for row_i in row.split(":"):
            base = row[1:-1]
            if len(base)==0:
                return np.nan
            
            wt, idx, mt = row_i[0], int(row_i[1:-1]) - offset_idx, row_i[-1]
            if idx>=len(sequence):
               return np.nan
            
            assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
            # modify the sequence
            sequence = sequence[:idx] + mt + sequence[(idx + 1) :]
     
    if "_" in sequence:
        return np.nan #skip all delition and insertion
    
    # encode the sequence
    data = [
        ("protein1", sequence),
    ]
    
    batch_converter = alphabet.get_batch_converter()
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # compute probabilities at each position
    mt_representation = model(batch_tokens.cuda(),repr_layers=[model.num_layers])["representations"][model.num_layers].squeeze(0)
    #print(mt_representation.shape)
    if mode=="whole":
       #mask = (mt_representation != alphabet.padding_idx)  # use this in v2 training
       #denom = torch.sum(mask, -1, keepdim=True)
       #mt_representation = torch.sum(mt_representation * mask, dim=1) / denom  # remove padding
       #wt_representation = torch.sum(wt_representation * mask, dim=1) / denom
       mt_representation = torch.sum(mt_representation,dim=0)
       wt_representation = torch.sum(wt_representation,dim=0)
       
    elif mode =="marginals":   #add BOS at beginning
        mt_representation = mt_representation[idx+1:idx+2].squeeze(0)
        wt_representation = wt_representation[idx+1:idx+2].squeeze(0)
    elif mode == "RLA":
        if  similarity == "cosine":    
           score = (mt_representation.unsqueeze(0).unsqueeze(2) @ wt_representation.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(-1).squeeze(0)
           return (score).mean(0).item()
        elif similarity == "euclidean_distance":
           score = np.linalg.norm((mt_representation-wt_representation).to('cpu').detach().numpy(),axis=1)
           return np.log(np.mean(score))*-1
        elif similarity == "mse":
           score = np.mean(((mt_representation - wt_representation).to('cpu').detach().numpy()) ** 2)
           return np.log(score)*-1
    
    #print(mt_representation.shape)
    #print(wt_representation.shape)
    if similarity == "cosine":
       score = F.cosine_similarity(mt_representation.unsqueeze(0), wt_representation.unsqueeze(0))
    elif similarity == "euclidean_distance":
       score = torch.dist(mt_representation, wt_representation, p=2)
    
    elif similarity == "correlation":
       # Stack the tensors into a 2D tensor
       stacked_tensors = torch.stack((mt_representation, wt_representation))
       # Calculate the Pearson correlation coefficient matrix
       correlation_matrix = torch.corrcoef(stacked_tensors)
       score = correlation_matrix[0, 1]
    
    return np.log(score.item())

def test_DMS(configs, seq_model, alphabet, n_steps, logging):
    # parser = create_parser()
    # args = parser.parse_args()
    # args.model,args.alphabet = load_model(args)
    # outputfilename = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/ESM_1v_data/results/",'ESM-1v_data_summary_start_end_seq_splm_esm2.csv')
    outputfilename = os.path.join(configs.valid_settings.dms_summary_path)
    dfdata=pd.read_csv(outputfilename)
    tqdm.pandas()
    for index, row in dfdata.iterrows():
      #for index, row in df.iloc[index:].iterrows():
        wt_seq = row['WT seq']
        if wt_seq is np.nan:
            continue
        
        filekey = row['Dataset_file']
        ### just for mutation effect to stability
        # print(args.filekey)
        if not filekey == 'PTEN_HUMAN_Fowler2018':
        # if not filekey == 'TPMT_HUMAN_Fowler2018' and not filekey == 'PTEN_HUMAN_Fowler2018':
            # if args.scoring_strategy in ["wt-mt-whole","wt-mt-RLA","wt-mt-marginals"]:
            #    df.loc[index,args.modelname+"-"+args.scoring_strategy+"-"+args.similarity] = 0
            # else:
            #     df.loc[index,args.modelname+"-"+args.scoring_strategy] = 0
            continue
        # path_test = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/ESM_1v_data/data_wedownloaded/41 mutation dataset/ESM-1v-41/test",row['Dataset_file']+".csv")
        # path_val = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/ESM_1v_data/data_wedownloaded/41 mutation dataset/ESM-1v-41/validation",row['Dataset_file']+".csv")
        path_test = os.path.join(configs.valid_settings.dms_path, 'test', row['Dataset_file']+".csv")
        path_val = os.path.join(configs.valid_settings.dms_path, 'validation', row['Dataset_file']+".csv")
        if os.path.exists(path_test):
           dms_input = path_test
        else:
           dms_input = path_val
        
        sequence = str(wt_seq)
        offset_idx = int(row['offset_idx'])
        ref_name = row['Name(s) in Reference']
        # result = main(args)

        df = pd.read_csv(dms_input)
        batch_converter = alphabet.get_batch_converter()
        data = [
        ("protein1", sequence),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        mode = "RLA"
        tqdm.pandas()
        with torch.no_grad():
            wt_representation = seq_model(batch_tokens.cuda(),repr_layers=[seq_model.num_layers])["representations"][seq_model.num_layers]
        
        wt_representation = wt_representation.squeeze(0) #only one sequence a time
        mutation_col = "mutant"
        similarity = "euclidean_distance"
        df['modelname'] = df.progress_apply(
            lambda row: compute_repdiff(
                row[mutation_col],
                sequence,
                seq_model,
                alphabet,
                offset_idx,
                wt_representation,
                mode=mode,
                similarity=similarity
            ),
            axis=1,
        )
        esm_rla_corr = df[ref_name].corr(df['modelname'],method = 'pearson')
        esm_rla_spearmn = df[ref_name].corr(df['modelname'],method='spearman')  
    
    
    # logging.info(f'evaluation - step {n_steps}')
    logging.info(f"step:{n_steps} esm_rla_spearmn:{esm_rla_spearmn:.4f}")

    return esm_rla_spearmn

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

def test_rmsf_cor(val_loader, alphabet, configs, seq_model, n_steps, logging):
    processed_list=[]
    rep_norm_list=[]
    rmsf_list=[]
    for batch in val_loader:
        pid = batch['pid'][0]
        pid = pid.split('#')[0]
        if pid in processed_list:
            continue
        pdb_file = os.path.join(configs.valid_settings.analysis_path, f"{pid}_analysis", f"{pid}.pdb")
        # pdb_file = os.path.join("../analysis/", f"{pid}_analysis", f"{pid}.pdb")
        sequence = str(pdb2seq(pdb_file))
        data = [("protein1", sequence),]
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            wt_representation = seq_model(batch_tokens.cuda(),repr_layers=[seq_model.num_layers])["representations"][seq_model.num_layers]
        
        wt_representation = wt_representation.squeeze(0) #only one sequence a time
        seq_emb = wt_representation[1:-1]
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        rmsf_file = os.path.join(configs.valid_settings.analysis_path, f"{pid}_analysis", f"{pid}_RMSF.tsv")
        df = pd.read_csv(rmsf_file, sep="\t")
        r1 = df["RMSF_R1"].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r1)
        processed_list.append(pid)
        # print(pid)
        # print(len(sequence))
        # print(len(residue_norms))
        # print(len(r1))
    
    corr, _ = spearmanr(rep_norm_list, rmsf_list)
    logging.info(f"step:{n_steps} rmsf_cor:{corr:.4f}")
    return corr

def test_rmsf_cor_mdcath(alphabet, configs, seq_model, n_steps, logging):
    processed_list=[]
    rep_norm_list=[]
    rmsf_list=[]
    num=0
    for subfolder in os.listdir(configs.valid_settings.analysis_path):
        if num>500:
            break
        pid=os.path.basename(subfolder)
        pid="_".join(pid.split("_")[:2])
        pdb_file = os.path.abspath(os.path.join(configs.valid_settings.analysis_path,
                                                subfolder, f"{pid}.pdb"))
        sequence = str(pdb2seq(pdb_file))
        data = [("protein1", sequence),]
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            wt_representation = seq_model(batch_tokens.cuda(),repr_layers=[seq_model.num_layers])["representations"][seq_model.num_layers]
        
        wt_representation = wt_representation.squeeze(0) #only one sequence a time
        seq_emb = wt_representation[1:-1]
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        rmsf_file = os.path.abspath(os.path.join(configs.valid_settings.analysis_path,
                                                 subfolder, f"{pid}_RMSF.tsv"))
        df = pd.read_csv(rmsf_file, sep="\t")
        r1 = df["RMSF_R1"].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r1)
        processed_list.append(pid)
        num+=1
    corr, _ = spearmanr(rep_norm_list, rmsf_list)
    logging.info(f"step:{n_steps} rmsf_cor:{corr:.4f}")
    return corr


# test_evaluate_allcases()
"""
#ESM2 650M： 
pool_mode=2
ARI in 2D dimention:
step: 1 digit_num_1:13.428166697543922(6.666182560121344)       digit_num_2:7.4104153527649546(6.516220669062293        digit_num_3:4.3119577929493325(3.7271170336459147)
step:1  digit_num_1_ARI:0.005696861722158926    digit_num_2_ARI:0.012342352314497188    digit_num_3_ARI:0.03680253448677986
step: 1 deaminase_score:14.851235426527973(70.8537540077077) ARI:0.6257119410672616
step: 1 kinase_score:10.01905618755323(16.818083083910178) ARI:0.27798039262981467

step: 1 deaminase_score:14.851238117106023(71.52366964524673) ARI:0.619341259186361
step: 1 kinase_score:10.019058316038905(15.438954033770596) ARI:0.2558232661978476

For original dimention:
step: 1 deaminase_score:14.851238117106023(71.52366964524673) ARI:0.5379895530581004
step: 1 kinase_score:10.019058316038905(15.438954033770596) ARI:0.21424176517590296

pool_mode=3 poorer
step: 1 digit_num_1:11.41722498614794(9.740341802665275)        digit_num_2:9.853495640984107(12.101065093032561digit_num_3:6.024049738288263(7.466253686809394)
step:1  digit_num_1_ARI:0.010995951037930525    digit_num_2_ARI:0.017665505642599007    digit_num_3_ARI:0.03228655684879297
step: 1 deaminase_score:11.518577904896345(36.8394533713406) ARI:0.6193089287923969
step: 1 kinase_score:6.997840746607161(7.095931242585982) ARI:0.22063111401642121


#finetune
#checkpoint_path="v3_tune_allembeddingtable_contactv2_fixswin/checkpoint_0174174.pth.tar"
step: 1 digit_num_1:5.5748190792687025(18.361988528460987)      digit_num_2:2.0214611707195296(3.670287400171874)
step: 1 deaminase_score:19.540550462058963(97.14474700511767)
step: 1 kinase_score:31.60880092458882(189.4671489823389)

#adapterH 
#checkpoint_path="./v3_adapterH_tune_esmembeddingtable_contactv2_fixswincorrect/checkpoint_0160160.pth.tar"
step: 1 digit_num_1:67.10818137290548(320.61274163755206)       digit_num_2:10.630953560143524(45.91206233326133)
step: 1 deaminase_score:10.718848846608283(34.64923076803777)
step: 1 kinase_score:14.626499161959586(65.79183430060112)

#checkpoint_path="./config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3/checkpoints/checkpoint_0162000.pth"
#kmeans ari: 0.85 
#checkpoint_path = checkpoint_path="./config_16fixesm_sgdlr001mixlenpad_esmpool2_MLMcontrast_testnolmheadbz20_run2/saved_checkpoints/checkpoint_0520000.pth"
#ARI in original dimension
step: 1 deaminase_score:21.70997731142161(137.4967469057334) ARI:0.8209265281045667
step: 1 kinase_score:39.421032703607544(396.40919127358666) ARI:0.576863315122049
#ARI in 2D
step: 1 deaminase_score:21.70997731142161(137.4967469057334) ARI:0.8648526899736879
step: 1 kinase_score:39.421032703607544(396.40919127358666) ARI:0.7690686582248041

"

"""
