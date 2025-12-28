import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import argparse
import torch
import esm
import esm_adapterH
import yaml
from utils.utils import load_configs
from collections import OrderedDict
from peft import LoraConfig, get_peft_model


import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

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
# ----------------------------
# Setup
# ----------------------------
parser = argparse.ArgumentParser(description="DMS regression tasks")
parser.add_argument('--model_type', type=str, required=False, default="s-plm")
parser.add_argument('--config_path', type=str, required=False, default="../results/vivit_2/config.yaml")
parser.add_argument('--model_location', type=str, required=False, 
                    default="../results/vivit_2/checkpoints/checkpoint_best_val_dms_corr.pth")



args = parser.parse_args(args=['--model_type', "s-plm", #"s-plm" "ESM2"
                               '--config_path', "./results/vivit_3/config_vivit3.yaml",
                               '--model_location', "./results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth"]
                               )


%matplotlib inline
def embed_sequence(mt_seq, wt_seq,batch_converter, args, pos):
    data = [
        ("protein1", mt_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        mt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    mt_representation = mt_representation.squeeze(0) #only one sequence a time
    data = [
        ("protein1", wt_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        wt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    wt_representation = wt_representation.squeeze(0) #only one sequence a time

    mutated_indices = [i for i, (a, b) in enumerate(zip(wt_seq, mt_seq)) if a != b]

    score_pro = np.linalg.norm((mt_representation-wt_representation).to('cpu').detach().numpy(),axis=1)
    score = np.mean(score_pro)
    for pos in mutated_indices:
        if pos==0 or pos==len(wt_seq)-1:
            score_pos = np.linalg.norm((mt_representation[pos+1]-wt_representation[pos+1]).to('cpu').detach().numpy())
        else:
            score_pos = np.linalg.norm((mt_representation[pos+1-1:pos+1+2]-wt_representation[pos+1-1:pos+1+2]).to('cpu').detach().numpy(),axis=1)
        score+=np.mean(score_pos)

    return np.log(score)*-1

def mask_marginal_score(mut_seq, seq, batch_converter, args, pos):
    if len(seq) != len(mut_seq):
        raise ValueError("Sequences must have the same length for direct comparison.")
    
    mutated_indices = [i for i, (a, b) in enumerate(zip(seq, mut_seq)) if a != b]

    data = [
        ("protein1", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    score=0
    for pos in mutated_indices:
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, pos+1] = args.alphabet.mask_idx
        with torch.no_grad():        
                token_probs = torch.log_softmax(        
                    args.model(batch_tokens_masked.cuda())["logits"], dim=-1        
                )
        token_probs = token_probs[:,pos+1]
        wt_encoded, mt_encoded = args.alphabet.get_idx(seq[pos]), args.alphabet.get_idx(mut_seq[pos])
        score += (token_probs[0, mt_encoded] - token_probs[0, wt_encoded])
    
    score = score.to('cpu').detach().numpy()
    return score/len(mutated_indices)

    # data = [
    #     ("protein1", seq),
    # ]
    # batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_tokens_masked = batch_tokens.clone() 
    # batch_tokens_masked[0, pos+1] = args.alphabet.mask_idx
    # with torch.no_grad():        
    #         token_probs = torch.log_softmax(        
    #             args.model(batch_tokens_masked.cuda())["logits"], dim=-1        
    #         )
    # token_probs = token_probs[:,pos+1]
    # wt_encoded, mt_encoded = args.alphabet.get_idx(seq[pos]), args.alphabet.get_idx(mut_seq[pos])
    # score = token_probs[0, mt_encoded] - token_probs[0, wt_encoded]
    # return score.to('cpu').detach().numpy()

def apply_mutation(seq, mutation):
    """
    Apply mutation like 'Q2C' on seq.
    """
    ref = mutation[0]
    pos = int(mutation[1:-1]) - 1  # convert to 0-index
    alt = mutation[-1]
    assert seq[pos] == ref, f"Reference mismatch at pos {pos+1}: expected {ref}, got {seq[pos]}"
    mutated = seq[:pos] + alt + seq[pos+1:]
    return mutated

def load_checkpoints(model,checkpoint_path):
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            if  any('adapter' in name for name, _ in model.named_modules()):
                if np.sum(["adapter_layer_dict" in key for key in checkpoint[
                    'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
                    new_ordered_dict = OrderedDict()
                    for key, value in checkpoint['state_dict1'].items():
                        if "adapter_layer_dict" not in key:
                            new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                            new_ordered_dict[new_key] = value
                        else:
                            new_ordered_dict[key] = value
                    
                    model.load_state_dict(new_ordered_dict, strict=False)
                else:
                    #this model does not contain esm2
                    new_ordered_dict = OrderedDict()
                    for key, value in checkpoint['state_dict1'].items():
                            key = key.replace("esm2.","")
                            new_ordered_dict[key] = value
                    
                    model.load_state_dict(new_ordered_dict, strict=False)
            elif  any('lora' in name for name, _ in model.named_modules()):
                #this model does not contain esm2
                new_ordered_dict = OrderedDict()
                for key, value in checkpoint['state_dict1'].items():
                        print(key)
                        key = key.replace("esm2.","")
                        new_ordered_dict[key] = value
                
                model.load_state_dict(new_ordered_dict, strict=False)
            
            print("checkpoints were loaded from " + checkpoint_path)
        else:
            print("checkpoints not exist "+ checkpoint_path)

def load_model(args):
    if args.model_type=="s-plm":
        with open(args.config_path) as file:
            config_file = yaml.full_load(file)
            configs = load_configs(config_file, args=None)
        if configs.model.esm_encoder.adapter_h.enable:
            model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)
        elif configs.model.esm_encoder.lora.enable:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            lora_targets =  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj","self_attn.out_proj"]
            target_modules=[]
            if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                start_layer_idx = np.max([model.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                for idx in range(start_layer_idx, model.num_layers):
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
            peft_model = get_peft_model(model, peft_config)
        
        # inference for each model
        # model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)/
        load_checkpoints(model,args.model_location)
    elif args.model_type=="ESM2":
        #if use ESM2
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    
    return model,alphabet

args.model, args.alphabet = load_model(args)
# batch_converter = args.alphabet.get_batch_converter(truncation_seq_length=512)
batch_converter = args.alphabet.get_batch_converter()

def evaluate_with_cath_more(out_figure_path, steps, batch_size, model, batch_converter, cathpath):
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
            features_seq = model(batch_tokens.cuda(),repr_layers=[model.num_layers])["representations"][model.num_layers]
            features_seq = features_seq.squeeze(0)
            features_seq = features_seq.mean(axis=0) 
            features_seq = features_seq.reshape(1,-1)
            # print(features_seq.shape)
            # batch_tokens = batch_tokens.to(accelerator.device)
            # _, _, features_seq, _ = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
            seq_embeddings.extend(features_seq.cpu().detach().numpy())

    seq_embeddings = np.asarray(seq_embeddings)
    print(seq_embeddings.shape)
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

def evaluate_with_deaminase(out_figure_path, model, steps, batch_size, batch_converter, deaminasepath):
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
            features_seq = model(batch_tokens.cuda(),repr_layers=[model.num_layers])["representations"][model.num_layers]
            features_seq = features_seq.squeeze(0)
            features_seq = features_seq.mean(axis=0) 
            features_seq = features_seq.reshape(1,-1)
            # batch_tokens = batch_tokens.to(accelerator.device)
            # _, _, features_seq, _ = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
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

def evaluate_with_kinase(out_figure_path, steps, batch_converter, batch_size, kinasepath, model):
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
            features_seq = model(batch_tokens.cuda(),repr_layers=[model.num_layers])["representations"][model.num_layers]
            features_seq = features_seq.squeeze(0)
            features_seq = features_seq.mean(axis=0) 
            features_seq = features_seq.reshape(1,-1)
            # batch_tokens = batch_tokens.to(accelerator.device)
            # _, _, features_seq, _ = model(batch_tokens=batch_tokens, mode='sequence', return_embedding=True)
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

###########  cath, deaminase, kinase structure cluster ############
scores_cath = evaluate_with_cath_more(
        out_figure_path="./results/testx",
        steps=1,
        batch_size=1, #int(configs.train_settings.batch_size / 2),
        cathpath="./results/testx/Rep_subfamily_basedon_S40pdb.fa",
        model=model,
        batch_converter=batch_converter
    )

scores_deaminase = evaluate_with_deaminase(
        out_figure_path="./results/testx",
        steps=1,
        batch_size=1, #int(configs.train_settings.batch_size / 2),
        deaminasepath="./results/testx/Table_S1.xlsx",
        model=model,
        batch_converter=batch_converter
    )

scores_kinase = evaluate_with_kinase(out_figure_path="./results/testx",
                                         steps=1,
                                         batch_size=1, #int(configs.train_settings.batch_size / 2),
                                         kinasepath="./results/testx/GPS5.0_homo_hasPK_with_kinasedomain.txt",
                                         model=model,
                                         batch_converter=batch_converter
                                         )

import matplotlib.pyplot as plt
import numpy as np

# === Data ===
groups = ["Class", "Architecture", "Topology"]
esm2       = [13, 7, 4]
promptprot = [37, 19, 11]
prostt5     = [55, 21, 11]
dplm       = [61, 31, 16]

methods = ["ESM2", "PromptProtein", "ProstT5", "DPLM"]
all_methods = [esm2, promptprot, prostt5, dplm]

# === Plot settings ===
x = np.arange(len(groups))        # positions for groups
width = 0.2                       # width of each bar

fig, ax = plt.subplots(figsize=(8, 5))

colors=["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
# Plot bars for each method, shifted along x
ax.bar(x - 1.5*width, esm2, width, label="ESM2", color=colors[0])
ax.bar(x - 0.5*width, promptprot, width, label="PromptProtein", color=colors[1])
ax.bar(x + 0.5*width, prostt5, width, label="ProstT5", color=colors[2])
ax.bar(x + 1.5*width, dplm, width, label="DPLM", color=colors[3])

# === Formatting ===
ax.set_ylabel("Calinski Harabasz Index")
ax.set_xlabel("Groups")
ax.set_title("CATH structural hierarchy")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend(title="Method")

# Optional: add numeric labels
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.0f', padding=2)

plt.tight_layout()
plt.show()
# If running in non-interactive mode, use:
# plt.savefig("grouped_barplot.png", dpi=300, bbox_inches="tight")

import matplotlib.pyplot as plt

# === Data ===
methods = ["ESM2", "PromptProtein", "ProstT5", "DPLM"]
values = [0.63, 0.46, 0.80, 0.79]

# === Plot ===
plt.figure(figsize=(6, 4))
bars = plt.bar(methods, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], width=0.6)

# Add labels and title
plt.ylabel("ARI")
plt.title("Deaminase groups")

# Add numerical value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10)

# Optional: tidy look
plt.ylim(0, 1.0)
plt.tight_layout()

plt.show()
# If running in non-interactive mode, use:
# plt.savefig("method_barplot.png", dpi=300, bbox_inches="tight")

import matplotlib.pyplot as plt

# === Data ===
methods = ["ESM2", "PromptProtein", "ProstT5", "DPLM"]
values = [0.28, 0.28, 0.61, 0.72]

# === Plot ===
plt.figure(figsize=(6, 4))
bars = plt.bar(methods, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], width=0.6)

# Add labels and title
plt.ylabel("ARI")
plt.title("Kinase groups")

# Add numerical value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10)

# Optional: tidy look
plt.ylim(0, 1.0)
plt.tight_layout()

plt.show()

###################### PIN1 mutation stability ############
def plot_and_fit(x, y, xlabel, ylabel, correlation, output_path, figure_name):
    x = np.array(x)
    y = np.array(y)
    
    # Fit a linear regression line: y = m*x + b
    m, b = np.polyfit(x, y, 1)
    fit_line = m * x + b
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, fit_line, color='red', label=f'Fit Line: y = {m:.2f}x + {b:.2f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Spearman cor: {correlation}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_path, f"{figure_name}.png"))

wtdg = -1.565
file2ddg={'L7A': 0.158, 'P9A': -1.376, 'E12A': -1.541, 'K13A': 0.066, 'R14A': -0.975, 'M15A': -1.123, 'S16A': -1.56, 'R17A':-1.598,
          'S18A': -0.920, 'R21A': -1.541, 'V22A': -1.309, 'Y23A': 0.485, 'F25A': 0.571, 'H27A': -1.443, 'I28A': -1.206,
          'T29A': -0.32, 'S32A': -1.302, 'W34A': -1.161, 'E35A': -0.836}

seq = 'MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNSSSGGKNGQGEPARVRCSHLLVKHSQSRRPSSWRQEKITRTKEEALELINGYIQKIKSGEEDFESLASQFSDCSSAKARGDLGAFSRGQMQKPFEDASFALRTGEMSGPVFTDSGIHIILRTE'

ddg_list=[]
emb_list=[]
for mut in file2ddg.keys():
    mut_seq=apply_mutation(seq, mut)
    pos = int(mut[1:-1]) - 1
    embedding = embed_sequence(mut_seq, seq, batch_converter, args, pos)
    # emb_norm = np.linalg.norm(embedding)
    # emb_norm = mask_marginal_score(mut_seq, seq, batch_converter, args, pos)
    ddg_list.append(file2ddg[mut]-wtdg)
    emb_list.append(embedding)

corr, _ = spearmanr(ddg_list, emb_list)
args.output_path='./Pin1_ddt.png'
plot_and_fit(ddg_list, emb_list, 'rep_norm', 'rmsf', corr, args.output_path, 'rep_norm_rmsf_e')


########################  ddt data ##########################

ddg_list=[]
emb_list=[]
df = pd.read_csv("../evaluate/ddt/M1261.csv") #S669, S8754, M1261
# df = df.iloc[:,:3]
# df.columns = ["A", "mutation", "effect"] 
# print("Generating embeddings...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    wt_seq = row['wt_seq']
    mut_seq = row['mut_seq']
    effect = row['ddG']
    if pd.isna(effect) or effect == '':
        continue
    embedding = embed_sequence(mut_seq, wt_seq, batch_converter, args, pos)
    # emb_norm = np.linalg.norm(embedding)
    # emb_norm = mask_marginal_score(mut_seq, wt_seq, batch_converter, args, 666)
    ddg_list.append(effect)
    emb_list.append(embedding)

corr, _ = spearmanr(ddg_list, emb_list)
args.output_path='./Pin1_ddt.png'
plot_and_fit(ddg_list, emb_list, 'rep_norm', 'rmsf', corr, args.output_path, 'rep_norm_rmsf_e')

#################### combine score ################
import numpy as np
import pandas as pd
from scipy.stats import norm

pro_emb_dplm_score_list=[]
esm2_list=[]
df = pd.read_csv("../evaluate/ddt/S8754.csv") #S669, S8754, M1261
# df = df.iloc[:,:3]
# df.columns = ["A", "mutation", "effect"] 
# print("Generating embeddings...")


for _, row in tqdm(df.iterrows(), total=len(df)):
    wt_seq = row['wt_seq']
    mut_seq = row['mut_seq']
    effect = row['ddG']
    if pd.isna(effect) or effect == '':
        continue
    embedding = embed_sequence(mut_seq, wt_seq, batch_converter, args, pos)
    emb_norm = np.linalg.norm(embedding)
    pro_emb_dplm_score_list.append(emb_norm)
    emb_norm = mask_marginal_score(mut_seq, wt_seq, batch_converter, args, 666)
    esm2_list.append(emb_norm)
    # ddg_list.append(effect)
    # emb_list.append(emb_norm)


def zero_shot_stouffer(esm2_list, dplm_list):
    # df has columns: ['pos','mut','esm2_lr','dplm'] for a single protein
    # x_evo = -esm2_list.to_numpy()   # larger = worse
    # x_dyn = dplm_list.to_numpy()       # larger = worse
    x_evo = -np.array(esm2_list)
    x_dyn = np.array(dplm_list)

    # empirical right-tail quantiles (mid-ranks)
    def quantiles(x):
        r = x.argsort().argsort() + 1   # 1..N ascending
        q = (r - 0.5) / len(x)          # left-tail; but x already aligned so larger=worse
        return q

    q_evo = quantiles(x_evo)
    q_dyn = quantiles(x_dyn)

    # map to normal scores
    z_evo = norm.ppf(q_evo)
    z_dyn = norm.ppf(q_dyn)

    s = (z_evo + z_dyn) / np.sqrt(2)
    return pd.Series(s, index=df.index, name='S_zero_shot')

################# new emb score ################


def embed_seq2rep(mt_seq, wt_seq,batch_converter, args, pos):
    data = [
        ("protein1", mt_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        mt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    mt_representation = mt_representation.squeeze(0) #only one sequence a time
    data = [
        ("protein1", wt_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        wt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    wt_representation = wt_representation.squeeze(0) #only one sequence a time

    mutated_indices = [i for i, (a, b) in enumerate(zip(wt_seq, mt_seq)) if a != b]

    score_pro = np.mean((mt_representation-wt_representation).to('cpu').detach().numpy(),axis=0)
    rep = score_pro
    for pos in mutated_indices:
        if pos==0 or pos==len(wt_seq)-1:
            score_pos = np.mean((mt_representation[pos+1]-wt_representation[pos+1]).to('cpu').detach().numpy())
        else:
            score_pos = np.mean((mt_representation[pos+1-1:pos+1+2]-wt_representation[pos+1-1:pos+1+2]).to('cpu').detach().numpy(),axis=0)
        rep+=score_pos

    return rep


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, mean_absolute_percentage_error)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CatBoostRegressionModel:
    """
    CatBoost regression model for continuous value prediction.
    
    Features:
    - Train on one dataset, test on another
    - Handles high-dimensional embeddings
    - Comprehensive evaluation metrics
    - Visualization tools
    """
    
    def __init__(self,
                 iterations=1000,
                 learning_rate=0.1,
                 depth=6,
                 l2_leaf_reg=3,
                 random_seed=42,
                 early_stopping_rounds=50,
                 verbose=100):
        """
        Initialize CatBoost regressor.
        
        Parameters:
        -----------
        iterations : int
            Maximum number of boosting iterations
        learning_rate : float
            Learning rate
        depth : int
            Depth of trees
        l2_leaf_reg : float
            L2 regularization coefficient
        random_seed : int
            Random seed for reproducibility
        early_stopping_rounds : int
            Stop if validation metric doesn't improve
        verbose : int
            Print progress every N iterations
        """
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            loss_function='RMSE',
            eval_metric='RMSE',
            task_type='CPU'  # Change to 'GPU' if you have GPU
        )
        
        self.feature_names = None
        self.fitted = False
        self.train_stats = None
    
    def load_data(self, filepath, feature_cols=None, label_col='label', 
                  embedding_col=None):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        feature_cols : list, optional
            List of column names to use as features
            If None, uses all columns except label_col
        label_col : str
            Name of the label column (default: 'label')
        embedding_col : str, optional
            If your embeddings are stored as a single column (e.g., string/list),
            specify the column name here
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Label vector
        df : pd.DataFrame
            Original dataframe
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples")
        
        # Extract labels
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV")
        
        y = df[label_col].values
        
        # Extract features
        if embedding_col is not None:
            # If embeddings are stored as strings like "[0.1, 0.2, ...]"
            print(f"Parsing embeddings from column '{embedding_col}'...")
            import ast
            X = np.array([ast.literal_eval(row) if isinstance(row, str) else row 
                         for row in df[embedding_col].values])
        elif feature_cols is not None:
            X = df[feature_cols].values
            self.feature_names = feature_cols
        else:
            # Use all columns except label
            feature_cols = [col for col in df.columns if col != label_col]
            X = df[feature_cols].values
            self.feature_names = feature_cols
        
        print(f"Feature shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Label statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
        
        return X, y, df
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            feature_names=None, cat_features=None):
        """
        Fit the CatBoost regression model.
        
        Parameters:
        -----------
        X_train : np.ndarray or pd.DataFrame
            Training features
        y_train : np.ndarray
            Training labels (continuous values)
        X_val : np.ndarray or pd.DataFrame, optional
            Validation features for early stopping
        y_val : np.ndarray, optional
            Validation labels
        feature_names : list, optional
            Names of features
        cat_features : list, optional
            Indices or names of categorical features
        """
        print(f"\n{'='*70}")
        print("TRAINING CATBOOST REGRESSOR")
        print(f"{'='*70}")
        print(f"Training samples: {len(y_train)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        # Set feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            if isinstance(X_train, pd.DataFrame):
                self.feature_names = X_train.columns.tolist()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Store training statistics for normalization info
        self.train_stats = {
            'y_mean': y_train.mean(),
            'y_std': y_train.std(),
            'y_min': y_train.min(),
            'y_max': y_train.max()
        }
        
        # Create Pool objects
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_features,
            feature_names=self.feature_names
        )
        
        eval_set = None
        if X_val is not None and y_val is not None:
            print(f"Validation samples: {len(y_val)}")
            val_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=cat_features,
                feature_names=self.feature_names
            )
            eval_set = val_pool
        
        # Fit model
        print("\nStarting training...")
        print("-"*70)
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            verbose=self.model.get_params()['verbose']
        )
        
        self.fitted = True
        print("-"*70)
        print("Training completed!")
        
        if eval_set is not None:
            print(f"Best iteration: {self.model.get_best_iteration()}")
            print(f"Best validation RMSE: {self.model.get_best_score()['validation']['RMSE']:.4f}")
    
    def predict(self, X):
        """Predict continuous values."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X, y, dataset_name="Test"):
        """
        Comprehensive evaluation of regression model.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features
        y : np.ndarray
            True labels
        dataset_name : str
            Name for the dataset
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate MAPE (handle zero values)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
        
        # Pearson correlation
        correlation = np.corrcoef(y, y_pred)[0, 1]
        
        # Residuals
        residuals = y - y_pred
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'correlation': correlation,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std()
        }
        
        # Print results
        print(f"\n{'='*70}")
        print(f"{dataset_name.upper()} SET PERFORMANCE")
        print(f"{'='*70}")
        print(f"RMSE:              {rmse:.4f}")
        print(f"MAE:               {mae:.4f}")
        print(f"R² Score:          {r2:.4f}")
        print(f"Pearson Corr:      {correlation:.4f}")
        print(f"MAPE:              {mape:.2f}%")
        print(f"MSE:               {mse:.4f}")
        print(f"Residual Mean:     {residuals.mean():.4f}")
        print(f"Residual Std:      {residuals.std():.4f}")
        print(f"{'='*70}\n")
        
        return metrics, y_pred
    
    def plot_predictions(self, y_true, y_pred, dataset_name="Test"):
        """
        Plot predicted vs actual values.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        dataset_name : str
            Dataset name for title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot: Predicted vs Actual
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        ax1.set_xlabel('True Value', fontsize=12)
        ax1.set_ylabel('Predicted Value', fontsize=12)
        ax1.set_title(f'Predicted vs Actual - {dataset_name} Set\nR²={r2:.4f}, Corr={correlation:.4f}', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Residual plot
        ax2 = axes[1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Value', fontsize=12)
        ax2.set_ylabel('Residual (True - Predicted)', fontsize=12)
        ax2.set_title(f'Residual Plot - {dataset_name} Set', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residual_distribution(self, y_true, y_pred, dataset_name="Test"):
        """Plot distribution of residuals."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Residual', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Residual Distribution - {dataset_name} Set\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}', 
                     fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Q-Q plot
        ax2 = axes[1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot - {dataset_name} Set', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_distribution(self, y_true, y_pred, dataset_name="Test"):
        """Plot absolute and percentage errors."""
        absolute_errors = np.abs(y_true - y_pred)
        percentage_errors = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute error vs true value
        ax1 = axes[0]
        ax1.scatter(y_true, absolute_errors, alpha=0.5, s=30)
        ax1.set_xlabel('True Value', fontsize=12)
        ax1.set_ylabel('Absolute Error', fontsize=12)
        ax1.set_title(f'Absolute Error vs True Value - {dataset_name} Set\nMAE={absolute_errors.mean():.4f}', 
                     fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Percentage error distribution
        ax2 = axes[1]
        ax2.hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Absolute Percentage Error (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Percentage Error Distribution - {dataset_name} Set\nMAPE={percentage_errors.mean():.2f}%', 
                     fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, top_k=20, plot=True):
        """
        Get and visualize feature importance.
        
        Parameters:
        -----------
        top_k : int
            Number of top features to show
        plot : bool
            Whether to create visualization
            
        Returns:
        --------
        importance_df : pd.DataFrame
            DataFrame with feature importance scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, max(6, top_k * 0.3)))
            
            top_features = importance_df.head(top_k)
            
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_k} Most Important Features', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return importance_df
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.model.load_model(filepath)
        self.fitted = True
        print(f"Model loaded from {filepath}")


# Main script: Train on one dataset, test on another
def main():
    """
    Example workflow: Train on one CSV, test on another CSV.
    """
    print("="*70)
    print("CATBOOST REGRESSION: TRAIN ON ONE DATASET, TEST ON ANOTHER")
    print("="*70)
    
    # File paths (MODIFY THESE TO YOUR ACTUAL FILES)
    train_file = "train_data.csv"  # Your training dataset
    test_file = "test_data.csv"    # Your test dataset
    
    # Initialize model
    model = CatBoostRegressionModel(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        early_stopping_rounds=50,
        verbose=100
    )
    
    # Load training data
    print("\n" + "="*70)
    print("LOADING TRAINING DATA")
    print("="*70)
    
    # Option 1: If your CSV has separate feature columns
    # X_train, y_train, train_df = model.load_data(
    #     train_file, 
    #     label_col='label'
    # )
    
    # Option 2: If embeddings are in a single column as string/list
    # X_train, y_train, train_df = model.load_data(
    #     train_file,
    #     label_col='label',
    #     embedding_col='embedding'
    # )
    
    # For demo: Generate synthetic data
    print("(Using synthetic data for demonstration)")
    n_train = 800
    n_test = 200
    n_features = 1280
    
    X_train = np.random.randn(n_train, n_features)
    y_train = np.sum(X_train[:, :50], axis=1) + np.random.randn(n_train) * 0.5
    
    X_test = np.random.randn(n_test, n_features)
    y_test = np.sum(X_test[:, :50], axis=1) + np.random.randn(n_test) * 0.5
    
    print(f"Training data: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    
    # Optional: Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    model.fit(
        X_train_split, y_train_split,
        X_val, y_val
    )
    
    # Evaluate on training data
    print("\n" + "="*70)
    print("EVALUATING ON TRAINING DATA")
    print("="*70)
    train_metrics, y_train_pred = model.evaluate(X_train, y_train, "Training")
    
    # Load and evaluate on test data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    
    # For real use: Load from CSV
    # X_test, y_test, test_df = model.load_data(
    #     test_file,
    #     label_col='label'
    # )
    
    print(f"Test data: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")
    
    # Evaluate on test data
    print("\n" + "="*70)
    print("EVALUATING ON TEST DATA")
    print("="*70)
    test_metrics, y_test_pred = model.evaluate(X_test, y_test, "Test")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Predictions vs actual
    model.plot_predictions(y_test, y_test_pred, "Test")
    
    # Residual distribution
    model.plot_residual_distribution(y_test, y_test_pred, "Test")
    
    # Error distribution
    model.plot_error_distribution(y_test, y_test_pred, "Test")
    
    # Feature importance
    print("\nAnalyzing feature importance...")
    importance_df = model.get_feature_importance(top_k=20, plot=True)
    
    # Save predictions
    print("\nSaving predictions...")
    results_df = pd.DataFrame({
        'true_value': y_test,
        'predicted_value': y_test_pred,
        'absolute_error': np.abs(y_test - y_test_pred),
        'percentage_error': np.abs((y_test - y_test_pred) / (y_test + 1e-10)) * 100
    })
    results_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'")
    
    # Save model
    model.save_model('catboost_regression_model.cbm')
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()


y=[]
X=[]
df = pd.read_csv("./evaluate/ddt/S8754.csv") #S669, S8754, M1261
# df = df.iloc[:,:3]
# df.columns = ["A", "mutation", "effect"] 
# print("Generating embeddings...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    wt_seq = row['wt_seq']
    mut_seq = row['mut_seq']
    effect = row['ddG']
    if pd.isna(effect) or effect == '':
        continue
    embedding = embed_seq2rep(mut_seq, wt_seq, batch_converter, args, pos)
    
    y.append(effect)
    X.append(embedding)

X_train=np.copy(X)
y_train=np.copy(y)
print(f"Training data: {X_train.shape}")
print(f"Training labels: {y_train.shape}")

y=[]
X=[]
df = pd.read_csv("./evaluate/ddt/S669.csv") #S669, S8754, M1261
# df = df.iloc[:,:3]
# df.columns = ["A", "mutation", "effect"] 
# print("Generating embeddings...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    wt_seq = row['wt_seq']
    mut_seq = row['mut_seq']
    effect = row['ddG']
    if pd.isna(effect) or effect == '':
        continue
    embedding = embed_seq2rep(mut_seq, wt_seq, batch_converter, args, pos)
    
    y.append(effect)
    X.append(embedding)

X_test=np.copy(np.array(X))
y_test=np.copy(np.array(y))

model = CatBoostRegressionModel(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        early_stopping_rounds=50,
        verbose=100
    )

from sklearn.model_selection import train_test_split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

# Train model
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

model.fit(
    X_train_split, y_train_split,
    X_val, y_val
)

# Evaluate on training data
print("\n" + "="*70)
print("EVALUATING ON TRAINING DATA")
print("="*70)
train_metrics, y_train_pred = model.evaluate(X_train, y_train, "Training")

# Load and evaluate on test data
print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

print(f"Test data: {X_test.shape}")
print(f"Test labels: {y_test.shape}")

# Evaluate on test data
print("\n" + "="*70)
print("EVALUATING ON TEST DATA")
print("="*70)
test_metrics, y_test_pred = model.evaluate(X_test, y_test, "Test")

################### soft spearman ddt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
log_filename = f'training_ddt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Still print to console
    ]
)
logger = logging.getLogger(__name__)

def extract_protein_id(name):
    """Extract protein ID from name: rcsb_1A7V_A_A104H_6_25 → 1A7V_A"""
    parts = name.split("_")
    return f"{parts[1]}_{parts[2]}"

class StabilityDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "protein_id": row["protein_id"],
            "wt_seq": row["wt_seq"],
            "mut_seq": row["mut_seq"],
            "ddg": torch.tensor(row["ddG"], dtype=torch.float32)
        }

class MLPHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def create_encoder_function(model, batch_converter):
    """
    Wrapper to create encoder function with your pretrained model.
    Assumes model and batch_converter are already loaded.
    """
    def encoder(seqs):
        batch_seq = [("seq_" + str(i), str(seqs[i])) for i in range(len(seqs))]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
        
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[model.num_layers])
            mt_representation = results["representations"][model.num_layers]
        
        # Average over sequence length: [B, L, dim] -> [B, dim]
        rep = torch.mean(mt_representation, axis=1)
        return rep
    
    return encoder

class StabilityModel(nn.Module):
    def __init__(self, encoder, embed_dim, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.mlp = MLPHead(embed_dim)
        self.freeze_encoder = freeze_encoder

    def encode(self, seqs):
        """Encode sequences using the pretrained encoder"""
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(seqs)
        else:
            return self.encoder(seqs)

    def forward(self, wt_seqs, mut_seqs):
        """
        Forward pass: encode both WT and mutant sequences,
        predict their scores, and return the difference (ddG prediction)
        """
        wt_emb = self.encode(wt_seqs)
        mut_emb = self.encode(mut_seqs)

        wt_score = self.mlp(wt_emb)
        mut_score = self.mlp(mut_emb)

        # Return difference: mut - wt
        return (mut_score - wt_score).squeeze(-1)

def soft_rank(x, tau=1.0):
    """Compute soft ranking using sigmoid approximation"""
    x = x.unsqueeze(0)  # (1, N)
    diff = x.T - x  # (N, N)
    P = torch.sigmoid(diff / tau)
    return P.sum(dim=1) + 1

def soft_spearman(pred, target, tau=1.0):
    """Compute differentiable Spearman correlation"""
    if pred.std() < 1e-6 or target.std() < 1e-6:
        # Return 0 correlation if no variance (will result in loss of 1.0, which is ignored)
        return torch.tensor(0.0, device=pred.device)
    
    pr = soft_rank(pred, tau)
    tr = soft_rank(target, tau)

    pr = pr - pr.mean()
    tr = tr - tr.mean()

    numerator = (pr * tr).sum()
    pr_norm = torch.sqrt((pr ** 2).sum())
    tr_norm = torch.sqrt((tr ** 2).sum())
    
    # Check for zero norms
    if pr_norm < 1e-6 or tr_norm < 1e-6:
        return torch.tensor(0.0, device=pred.device)
    
    denominator = pr_norm * tr_norm + 1e-8
    
    # Clamp to avoid numerical issues
    corr = numerator / denominator
    corr = torch.clamp(corr, -1.0, 1.0)
    return corr

def soft_spearman_loss(pred, target, tau=1.0):
    """Loss version: 1 - correlation"""
    corr = soft_spearman(pred, target, tau)
    # If correlation is 0 (invalid), return 0 loss so it doesn't contribute
    if torch.abs(corr) < 1e-6:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - corr

def train_step(model, batch, optimizer, lambda_rank=0.5, tau=0.5, chunk_size=32):
    """
    Training step with combined MSE and soft Spearman loss.
    Soft Spearman is computed per protein (only for proteins with 2+ mutations).
    """
    model.train()
    optimizer.zero_grad()

    wt, mut, ddg, prot_ids = batch
    # pred = model(wt, mut)
    batch_size = len(wt)
    if batch_size > chunk_size:
        all_preds = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            wt_chunk = wt[i:end_idx]
            mut_chunk = mut[i:end_idx]
            pred_chunk = model(wt_chunk, mut_chunk)
            all_preds.append(pred_chunk)
        
        pred = torch.cat(all_preds, dim=0)
    else:
        pred = model(wt, mut)

    # MSE loss over entire batch
    loss_mse = F.mse_loss(pred, ddg)

    # Soft Spearman loss per protein
    rank_losses = []
    unique_prots = set(prot_ids)

    for p in unique_prots:
        # Get indices for this protein
        mask = torch.tensor([pid == p for pid in prot_ids], device=pred.device)
        n_samples = mask.sum().item()
        
        if n_samples < 2:
            continue

        # Compute Spearman loss for this protein
        pred_p = pred[mask]
        ddg_p = ddg[mask]
        rank_losses.append(soft_spearman_loss(pred_p, ddg_p, tau))

    # Average Spearman loss across proteins
    if len(rank_losses) > 0:
        loss_rank = torch.stack(rank_losses).mean()
    else:
        loss_rank = torch.tensor(0.0, device=pred.device)

    # Combined loss
    loss = loss_mse + lambda_rank * loss_rank
    
    # Check for NaN before backward
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: NaN/Inf loss detected. MSE: {loss_mse.item()}, Rank: {loss_rank.item()}")
        # Skip this batch
        return {
            "total": 0.0,
            "mse": loss_mse.item() if not torch.isnan(loss_mse) else 0.0,
            "spearman": loss_rank.item() if not torch.isnan(loss_rank) else 0.0,
            "n_proteins_with_rank": len(rank_losses)
        }
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total": loss.item(),
        "mse": loss_mse.item(),
        "spearman": loss_rank.item() if len(rank_losses) > 0 else 0.0,
        "n_proteins_with_rank": len(rank_losses)
    }

@torch.no_grad()
def eval_step(model, dataloader):
    """Evaluation: compute average MSE"""
    model.eval()
    losses = []

    for batch in dataloader:
        wt, mut, ddg, prot_ids = batch
        ddg = ddg.cuda()
        pred = model(wt, mut)
        losses.append(F.mse_loss(pred, ddg).item())

    return sum(losses) / len(losses) if losses else 0.0

def collate_fn(batch):
    """Collate function for DataLoader"""
    return (
        [b["wt_seq"] for b in batch],
        [b["mut_seq"] for b in batch],
        torch.stack([b["ddg"] for b in batch]),
        [b["protein_id"] for b in batch],
    )

class ProteinBatchSampler(Sampler):
    """
    Custom sampler that returns all mutations from one protein per batch.
    This is crucial for computing per-protein Spearman correlation.
    """
    def __init__(self, dataset, shuffle=True):
        self.shuffle = shuffle
        self.protein_to_indices = defaultdict(list)

        for idx, row in enumerate(dataset.df.itertuples()):
            self.protein_to_indices[row.protein_id].append(idx)

        self.proteins = list(self.protein_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.proteins)

        for p in self.proteins:
            yield self.protein_to_indices[p]

    def __len__(self):
        return len(self.proteins)


df = pd.read_csv("./evaluate/ddt/S8754.csv")
df["protein_id"] = df["name"].apply(extract_protein_id)
# Split by protein (to avoid data leakage)
proteins = df["protein_id"].unique()
train_prots, val_prots = train_test_split(
    proteins, test_size=0.2, random_state=42
)
train_df = df[df["protein_id"].isin(train_prots)]
val_df = df[df["protein_id"].isin(val_prots)]
logger.info(f"Training proteins: {len(train_prots)}")
logger.info(f"Validation proteins: {len(val_prots)}")
logger.info(f"Training samples: {len(train_df)}")
logger.info(f"Validation samples: {len(val_df)}")
# Create datasets
train_dataset = StabilityDataset(train_df)
val_dataset = StabilityDataset(val_df)
# Create data loaders with protein-wise batching
train_loader = DataLoader(
    train_dataset,
    batch_sampler=ProteinBatchSampler(train_dataset, shuffle=True),
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=ProteinBatchSampler(val_dataset, shuffle=False),
    collate_fn=collate_fn
)

encoder = create_encoder_function(args.model, batch_converter)  # Replace with your encoder
embed_dim = 1280  # Replace with your embedding dimension
# Create model
model = StabilityModel(encoder, embed_dim, freeze_encoder=True).cuda()

# Optimizer (only train MLP head if encoder is frozen)
optimizer = torch.optim.Adam(model.mlp.parameters(), lr=1e-3)
# Training loop
num_epochs = 5
lambda_rank = 0.5  # Weight for Spearman loss
tau = 0.5  # Temperature for soft ranking

for epoch in range(1, num_epochs + 1):
    train_losses = []
    train_mse = []
    train_spearman = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        # Move ddG to GPU
        batch = (batch[0], batch[1], batch[2].cuda(), batch[3])
        loss_dict = train_step(model, batch, optimizer, 
                              lambda_rank=lambda_rank, tau=tau)
        
        train_losses.append(loss_dict["total"])
        train_mse.append(loss_dict["mse"])
        train_spearman.append(loss_dict["spearman"])
        # Update progress bar with current losses
        pbar.set_postfix({
            'loss': f'{loss_dict["total"]:.4f}',
            'mse': f'{loss_dict["mse"]:.4f}',
            'sp': f'{loss_dict["spearman"]:.4f}',
            'n_prot': loss_dict["n_proteins_with_rank"]
        })

        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Epoch {epoch} Batch {batch_idx+1}: "
                      f"Loss={loss_dict['total']:.4f}, "
                      f"MSE={loss_dict['mse']:.4f}, "
                      f"Spearman={loss_dict['spearman']:.4f}, "
                      f"N_proteins={loss_dict['n_proteins_with_rank']}")
                
    val_mse = eval_step(model, val_loader)
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_mse = sum(train_mse) / len(train_mse)
    avg_train_spearman = sum(train_spearman) / len(train_spearman)

    logger.info("="*60)
    logger.info(f"Epoch {epoch:03d} Summary:")
    logger.info(f"  Train Loss: {avg_train_loss:.4f}")
    logger.info(f"  Train MSE: {avg_train_mse:.4f}")
    logger.info(f"  Train Spearman: {avg_train_spearman:.4f}")
    logger.info(f"  Val MSE: {val_mse:.4f}")
    logger.info("="*60)

logger.info("\nTraining completed!")

model_path = 'ddt_model_'+log_filename.split(".")[0]+'.pt'
torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
###################### evaluate
import torch
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

@torch.no_grad()
def collect_predictions(model, dataloader):
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_targets = []
    all_proteins = []

    for batch in dataloader:
        wt, mut, ddt, prot_ids = batch
        ddt = ddt.to(device)

        preds = model(wt, mut)

        all_preds.append(preds.cpu())
        all_targets.append(ddt.cpu())
        all_proteins.extend(prot_ids)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    return all_preds, all_targets, all_proteins

def protein_wise_spearman(preds, targets, proteins):
    """
    Returns:
        dict: protein_id -> Spearman rho
        float: average Spearman across proteins
    """
    by_protein = defaultdict(list)

    for p, y_hat, y in zip(proteins, preds, targets):
        by_protein[p].append((y_hat, y))

    protein_rhos = {}

    for p, values in by_protein.items():
        if len(values) < 2:
            continue

        pred_p, true_p = zip(*values)
        rho, _ = spearmanr(pred_p, true_p)

        # Handle NaN (can happen if predictions are constant)
        if not np.isnan(rho):
            protein_rhos[p] = rho

    avg_rho = np.mean(list(protein_rhos.values())) if protein_rhos else np.nan

    return protein_rhos, avg_rho

def global_spearman(preds, targets):
    rho, _ = spearmanr(preds, targets)
    return rho

def evaluate_model(model, test_loader):
    preds, targets, proteins = collect_predictions(model, test_loader)

    # Protein-wise
    protein_rhos, avg_protein_rho = protein_wise_spearman(
        preds, targets, proteins
    )

    # Global
    global_rho = global_spearman(preds, targets)

    return {
        "protein_wise": protein_rhos,
        "avg_protein_spearman": avg_protein_rho,
        "global_spearman": global_rho
    }

df_test = pd.read_csv("./evaluate/ddt/S669.csv")
df_test["protein_id"] = df_test["name"].apply(extract_protein_id)
test_dataset = StabilityDataset(df_test)
test_loader = DataLoader(
    test_dataset, 
    batch_size=32,
    collate_fn=collate_fn
)

results = evaluate_model(model, test_loader)

logger.info(f"Average protein-wise Spearman: {results['avg_protein_spearman']:.3f}")
logger.info(f"Global Spearman: {results['global_spearman']:.3f}")
####################### pre encode version ##################
df = pd.read_csv("./evaluate/ddt/S8754.csv")
df["protein_id"] = df["name"].apply(extract_protein_id)
# Split by protein (to avoid data leakage)
proteins = df["protein_id"].unique()
train_prots, val_prots = train_test_split(
    proteins, test_size=0.2, random_state=42
)
train_df = df[df["protein_id"].isin(train_prots)]
val_df = df[df["protein_id"].isin(val_prots)]
logger.info(f"Training proteins: {len(train_prots)}")
logger.info(f"Validation proteins: {len(val_prots)}")
logger.info(f"Training samples: {len(train_df)}")
logger.info(f"Validation samples: {len(val_df)}")

x_train=[]
y_train=[]

encoder = create_encoder_function(args.model, batch_converter)
for _, row in tqdm(df.iterrows(), total=len(df)):
    wt_seq = row['wt_seq']
    mut_seq = row['mut_seq']
    effect = row['ddG']
    if pd.isna(effect) or effect == '':
        continue
    emb_wt = encoder(mut_seq)
    emb_mt = encoder(mut_seq)
    
    y_train.append(effect)
    x_train.append((emb_wt, emb_mt))

df = pd.read_csv("./evaluate/ddt/S669.csv")
x_test=[]
y_test=[]
for _, row in tqdm(df.iterrows(), total=len(df)):
    wt_seq = row['wt_seq']
    mut_seq = row['mut_seq']
    effect = row['ddG']
    if pd.isna(effect) or effect == '':
        continue
    emb_wt = encoder(mut_seq)
    emb_mt = encoder(mut_seq)
    
    y_test.append(effect)
    x_test.append((emb_wt, emb_mt))
class myModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # self.encoder = encoder
        self.mlp = MLPHead(embed_dim)
        # self.freeze_encoder = freeze_encoder

    def forward(self, wt_emb, mt_emb):
        """
        Forward pass: encode both WT and mutant sequences,
        predict their scores, and return the difference (ddG prediction)
        """
        # wt_emb = self.encode(wt_seqs)
        # mut_emb = self.encode(mut_seqs)

        wt_score = self.mlp(wt_emb)
        mut_score = self.mlp(mt_emb)

        # Return difference: mut - wt
        return (mut_score - wt_score).squeeze(-1)
    
def train_step(model, x, y, optimizer):
    model.train()
    optimizer.zero_grad()
    preds=[]
    for i in x:
        pred = model(i[0],i[1])
        preds.append(pred)
    
    loss_mse = F.mse_loss(torch.from_numpy(np.array(pred)), torch.from_numpy(np.array(y)))
    loss_mse.backward()
    optimizer.step()
    return {
        # "total": loss.item(),
        "mse": loss_mse.item()
    }
def eval_step(model, x, y):
    """Evaluation: compute average MSE"""
    model.eval()
    losses = []

    for i in x:
        pred = model(i[0],i[1])
        # preds.append(pred)
        losses.append(F.mse_loss(torch.from_numpy(np.array(pred)), torch.from_numpy(np.array(y))).item())

    return sum(losses) / len(losses) if losses else 0.0

embed_dim = 1280
model = myModel(embed_dim).cuda()
optimizer = torch.optim.Adam(model.mlp.parameters(), lr=1e-3)
num_epochs = 5

for epoch in range(num_epochs):
    train_mse = []
    # pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    loss_dict = train_step(model, x_train, y_train, optimizer)
    train_mse.append(loss_dict['mse'])
    print(f"MSE={loss_dict['mse']:.4f}")
    