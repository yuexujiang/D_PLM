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

parser.add_argument('--reverse', action='store_true')


args = parser.parse_args(args=['--model_type', "s-plm", #"s-plm" "ESM2"
                               '--config_path', "./results/vivit_3/config_vivit3.yaml",
                               '--model_location', "./results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth",
                               '--reverse']
                               )



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

    # score = np.linalg.norm((mt_representation-wt_representation).to('cpu').detach().numpy(),axis=1)
    # return np.log(np.mean(score))*-1

    # return mt_representation[0]-wt_representation[0]
    score = (mt_representation-wt_representation).to('cpu').detach().numpy()
    return np.mean(score, axis=0)

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
batch_converter = args.alphabet.get_batch_converter(truncation_seq_length=512)

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
    emb_norm = np.linalg.norm(embedding)
    ddg_list.append(file2ddg[mut]-wtdg)
    emb_list.append(emb_norm)

corr, _ = spearmanr(ddg_list, emb_list)
plot_and_fit(ddg_list, emb_list, 'rep_norm', 'rmsf', corr, args.output_path, 'rep_norm_rmsf_e')















# ----------------------------
# Define your embedding function
# ----------------------------

def embed_sequence(mt_seq, wt_seq,batch_converter, args):
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

    # score = np.linalg.norm((mt_representation-wt_representation).to('cpu').detach().numpy(),axis=1)
    # return np.log(np.mean(score))*-1

    return (mt_representation[0]-wt_representation[0]).to('cpu').detach().numpy()
    """
    Replace this with your actual embedding model.
    This dummy version returns a fixed-length random vector.
    """
    np.random.seed(hash(seq) % 2**32)  # deterministic randomness per sequence
    return np.random.rand(128)

# ----------------------------
# Mutate sequence + embed
# ----------------------------

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

# Generate embeddings for each mutant sequence


# Your original sequence
#SEQ = "MASTQ..."  # Replace with your full WT sequence

uniprot_id='RL401_YEAST' #
if uniprot_id=='RL401_YEAST':
    SEQ='MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGIIEPSLKALASKYNCDKSVCRKCYARLPPRATNCRKRKCGHTNQLRPKKKLK'
    df = pd.read_csv("../ESM-1v data/data_wedownloaded/41 mutation dataset/ESM-1v-41/test/RL401_YEAST_Bolon2014.csv")
    df = df.iloc[:,:3]

# Load your mutation dataset (adjust if your CSV has headers etc.)
# df = pd.read_csv("your_file.csv", header=None)
df.columns = ["A", "mutation", "effect"]  # mutation in column B, effect in column C

parser = argparse.ArgumentParser(description="DMS regression tasks")
parser.add_argument('--model_type', type=str, required=False, default="s-plm")
parser.add_argument('--config_path', type=str, required=False, default="../results/vivit_2/config.yaml")
parser.add_argument('--model_location', type=str, required=False, 
                    default="../results/vivit_2/checkpoints/checkpoint_best_val_dms_corr.pth")

# safe parse
args = parser.parse_args(args=['--model_type', "s-plm", 
                               '--config_path', "../results/vivit_2/config.yaml",
                               '--model_location', "../results/vivit_2/checkpoints/checkpoint_best_val_dms_corr.pth"])

args.model, args.alphabet = load_model(args)
batch_converter = args.alphabet.get_batch_converter()
    

X = []
y = []
print("Generating embeddings...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    mut = row['mutation']
    effect = row['effect']
    mutated_seq = apply_mutation(SEQ, mut)
    if mutated_seq==SEQ:
        continue
    embedding = embed_sequence(mutated_seq, SEQ, batch_converter, args)
    X.append(embedding)
    y.append(effect)

X = np.array(X)
y = np.array(y)

print("Any NaN in X_train?", np.isnan(X).any())
print("Any inf in X_train?", np.isinf(X).any())

# ----------------------------
# Train and Evaluate at various data splits
# ----------------------------

if X.ndim==1:
    X=X.reshape(-1,1)

splits = [0.8, 0.6, 0.4, 0.2]
# for train_frac in splits:
#     X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_frac, random_state=42)

#     model = Ridge()  # or any other regressor
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_val)
#     mse = mean_squared_error(y_val, y_pred)
#     r2 = r2_score(y_val, y_pred)

#     print(f"\n=== Split: {int(train_frac*100)}% train / {int((1-train_frac)*100)}% val ===")
#     print(f"R^2 score: {r2:.4f}")
#     print(f"MSE: {mse:.4f}")
# === Hyperparameters ===
INPUT_DIM = X.shape[1]         # your embedding dimension (e.g., 128)
HIDDEN_DIM = 64
EPOCHS = 200
LR = 1e-3
PATIENCE = 15
DROPOUT = 0.2

# === Normalize y ===
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# === Convert to tensors ===
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# === Train/Val split (e.g., 80/20) ===
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, train_size=0.2, random_state=42)

# === MLP Model ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# === Early Stopping ===
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    pred_train = model(X_train)
    loss = loss_fn(pred_train, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val)
        val_loss = loss_fn(pred_val, y_val).item()

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        break

# === Evaluation with best model ===
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    final_pred = model(X_val).numpy()
    final_true = y_val.numpy()
    # Undo y normalization
    final_pred = scaler_y.inverse_transform(final_pred)
    final_true = scaler_y.inverse_transform(final_true)

    r2 = r2_score(final_true, final_pred)
    mse = mean_squared_error(final_true, final_pred)

print(f"\n✅ Best model performance on validation set:")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")