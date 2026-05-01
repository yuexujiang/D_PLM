# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

import esm
import esm_adapterH

import numpy as np
import yaml
from utils.utils import load_configs
from collections import OrderedDict
from Bio.PDB import PDBParser, PPBuilder
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances
import os
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import MDAnalysis as mda


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from peft import PeftModel, LoraConfig, get_peft_model


# def load_checkpoints(model,checkpoint_path):
#         if checkpoint_path is not None:
#             checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#             if np.sum(["adapter_layer_dict" in key for key in checkpoint[
#                 'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
#                 new_ordered_dict = OrderedDict()
#                 for key, value in checkpoint['state_dict1'].items():
#                     if "adapter_layer_dict" not in key:
#                         new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
#                         new_ordered_dict[new_key] = value
#                     else:
#                         new_ordered_dict[key] = value
                
#                 model.load_state_dict(new_ordered_dict, strict=False)
#             else:
#                 #this model does not contain esm2
#                 new_ordered_dict = OrderedDict()
#                 for key, value in checkpoint['state_dict1'].items():
#                         key = key.replace("esm2.","")
#                         new_ordered_dict[key] = value
                
#                 model.load_state_dict(new_ordered_dict, strict=False)
            
#             print("checkpoints were loaded from " + checkpoint_path)
#         else:
#             print("checkpoints not exist "+ checkpoint_path)

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

# def load_model(args):
#     if args.model_type=="d-plm":
#         with open(args.config_path) as file:
#             config_file = yaml.full_load(file)
#             configs = load_configs(config_file, args=None)
        
#         # inference for each model
#         model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)
#         load_checkpoints(model,args.model_location)
#     elif args.model_type=="ESM2":
#         #if use ESM2
#         model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    
#     model.eval()
#     if torch.cuda.is_available():
#         model = model.cuda()
#         print("Transferred model to GPU")
    
#     return model,alphabet

def load_model(args):
    if args.model_type=="d-plm":
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

def main(args):
    batch_converter = args.alphabet.get_batch_converter()
    
    data = [
        ("protein1", args.sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        wt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    
    wt_representation = wt_representation.squeeze(0) #only one sequence a time
    return wt_representation




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


def load_h5_file_v2(file_path):
    with h5py.File(file_path, 'r') as f:
        rmsf = f['rmsf'][:]
        dccm = f['dccm'][:]
        mi = f['mi'][:]
        pid = f['pid'][()].decode('utf-8')
        seq = f['seq'][()].decode('utf-8')  # Decode from bytes to string
        # seq = f['seq']
        # pid = f['pid']
    
    return rmsf, dccm, mi, seq, pid

def flatten_upper(mat):
    return mat[np.triu_indices_from(mat, k=1)]

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

def random_adapter_each(args):
    import torch
    each_cor_mean=[]
    for layer_idx in range(17,33):
    #
        args.model,args.alphabet = load_model(args)
        # --- Random ablation: reinitialize all adapter layer weights ---
        
        # for layer in args.model.layers:
        #     if hasattr(layer, 'adapter_layer_dict'):
        #         for adapter_list in layer.adapter_layer_dict.values():
        #             for adapter in adapter_list:          # each ResMLP
        #                 for module in adapter.module:
        #                     if hasattr(module, 'weight'):
        #                         # torch.nn.init.kaiming_uniform_(module.weight)
        #                         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #                     if hasattr(module, 'bias') and module.bias is not None:
        #                         torch.nn.init.zeros_(module.bias)
        for i, layer in enumerate(args.model.layers):
            if i==layer_idx:
                if hasattr(layer, 'adapter_layer_dict') and layer.adapter_layer_dict is not None:
                    for adapter_list in layer.adapter_layer_dict.values():
                        for adapter in adapter_list:
                            for module in adapter.module:
                                if hasattr(module, 'weight'):
                                    # torch.nn.init.kaiming_uniform_(module.weight)
                                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                                if hasattr(module, 'bias') and module.bias is not None:
                                    torch.nn.init.zeros_(module.bias)
                    #
                    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
                    # datapath_MDfeature = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test_R1/'
                    processed_list=[]
                    rep_norm_list=[]
                    rmsf_list=[]
                    # rep_dis_list=[]
                    # dccm_list=[]
                    corr_list=[]
                    for file_name in os.listdir(datapath):
                        print(f"processing...{file_name}")
                        pid="_".join(file_name.split('_')[:2])
                        if pid in processed_list:
                            continue
                        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
                        args.sequence = str(pdb2seq(pdb_file))
                        #
                        seq_emb = main(args) #[seq_l+2, 1280]
                        seq_emb = seq_emb[1:-1]
                        # print(seq_emb.shape)
                        distance_matrix = pairwise_distances(seq_emb.cpu(), metric="euclidean")  # or cosine
                        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
                        #
                        rmsf_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_RMSF.tsv")
                        df = pd.read_csv(rmsf_file, sep="\t")
                        r1 = df["RMSF_R1"].values
                        # rep_norm_list.extend(residue_norms)
                        # rmsf_list.extend(r1)
                        corr, _ = spearmanr(residue_norms, r1)
                        corr_list.append(corr)
                        
                        processed_list.append(pid)
                    each_cor_mean.append(np.mean(corr_list))
    return each_cor_mean
    #
    # return (rep_norm_list, rmsf_list), (rep_dis_list, dccm_list)
    # return corr_list

def random_adapter(args):
    #
    args.model,args.alphabet = load_model(args)
    # --- Random ablation: reinitialize all adapter layer weights ---
    import torch
    # for layer in args.model.layers:
    #     if hasattr(layer, 'adapter_layer_dict'):
    #         for adapter_list in layer.adapter_layer_dict.values():
    #             for adapter in adapter_list:          # each ResMLP
    #                 for module in adapter.module:
    #                     if hasattr(module, 'weight'):
    #                         # torch.nn.init.kaiming_uniform_(module.weight)
    #                         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #                     if hasattr(module, 'bias') and module.bias is not None:
    #                         torch.nn.init.zeros_(module.bias)
    for layer in args.model.layers:
        if hasattr(layer, 'adapter_layer_dict') and layer.adapter_layer_dict is not None:
            for adapter_list in layer.adapter_layer_dict.values():
                for adapter in adapter_list:
                    for module in adapter.module:
                        if hasattr(module, 'weight'):
                            # torch.nn.init.kaiming_uniform_(module.weight)
                            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if hasattr(module, 'bias') and module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
    #
    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    # datapath_MDfeature = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test_R1/'
    processed_list=[]
    rep_norm_list=[]
    rmsf_list=[]
    # rep_dis_list=[]
    # dccm_list=[]
    corr_list=[]
    for file_name in os.listdir(datapath):
        file_name = file_name.strip()
        if not file_name:          # skip empty filenames
            continue
        print(f"processing...{file_name}")
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))
        #
        seq_emb = main(args) #[seq_l+2, 1280]
        seq_emb = seq_emb[1:-1]
        # print(seq_emb.shape)
        distance_matrix = pairwise_distances(seq_emb.cpu(), metric="euclidean")  # or cosine
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        #
        rmsf_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_RMSF.tsv")
        df = pd.read_csv(rmsf_file, sep="\t")
        r1 = df["RMSF_R1"].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r1)
        corr, _ = spearmanr(residue_norms, r1)
        corr_list.append(corr)
        
        processed_list.append(pid)
    #
    # return (rep_norm_list, rmsf_list), (rep_dis_list, dccm_list)
    return corr_list

def gogogo(args):
    #
    args.model,args.alphabet = load_model(args)
    #
    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    # datapath_MDfeature = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test_R1/'
    processed_list=[]
    rep_norm_list=[]
    rmsf_list=[]
    # rep_dis_list=[]
    # dccm_list=[]
    corr_list=[]
    for file_name in os.listdir(datapath):
        file_name = file_name.strip()
        if not file_name:          # skip empty filenames
            continue
        print(f"processing...{file_name}")
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))
        #
        seq_emb = main(args) #[seq_l+2, 1280]
        seq_emb = seq_emb[1:-1]
        # print(seq_emb.shape)
        distance_matrix = pairwise_distances(seq_emb.cpu(), metric="euclidean")  # or cosine
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        #
        rmsf_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_RMSF.tsv")
        df = pd.read_csv(rmsf_file, sep="\t")
        r1 = df["RMSF_R1"].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r1)
        corr, _ = spearmanr(residue_norms, r1)
        corr_list.append(corr)
        # r2 = df["RMSF_R2"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r2)
        # r3 = df["RMSF_R3"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r3)
        # print(r1.shape)
        # MDfeature_file_r1 = os.path.join(datapath_MDfeature, f"{file_name[:-3]}.h5")
        # if os.path.exists(MDfeature_file_r1):
        #     rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r1)
        #     r1 = flatten_upper(distance_matrix)
        #     r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
        #     rep_dis_list.extend(r1)
        #     dccm_list.extend(r2)
        #
        # MDfeature_file_r2 = os.path.join(datapath_MDfeature, f"{file_name}_prod_R2_fit.h5")
        # if os.path.exists(MDfeature_file_r2):
        #     rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r2)
        #     r1 = flatten_upper(distance_matrix)
        #     r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
        #     rep_dis_list.extend(r1)
        #     dccm_list.extend(r2)
        
        # MDfeature_file_r3 = os.path.join(datapath_MDfeature, f"{file_name}_prod_R3_fit.h5")
        # if os.path.exists(MDfeature_file_r3):
        #     rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r3)
        #     r1 = flatten_upper(distance_matrix)
        #     r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
        #     rep_dis_list.extend(r1)
        #     dccm_list.extend(r2)
        #
        # corr, _ = spearmanr(r1, r2)
        # corr_rmsf, _ = spearmanr(residue_norms, rmsf)
        processed_list.append(pid)
    #
    # return (rep_norm_list, rmsf_list), (rep_dis_list, dccm_list)
    return corr_list, (rep_norm_list, rmsf_list)


def compute_autocorr_times(positions):
    """
    Compute per-residue autocorrelation times from C-alpha trajectory.

    Parameters
    ----------
    positions : np.ndarray, shape [T, N, 3]
        C-alpha positions over T frames for N residues.

    Returns
    -------
    tau_c : np.ndarray, shape [N]
        Per-residue autocorrelation time in units of trajectory frames.
    """
    T, N, _ = positions.shape
    max_lag = T // 5  # reliable lag range: standard T/5 rule
    lags = np.arange(max_lag, dtype=np.float64)

    # Mean-center each residue across time
    delta = positions - positions.mean(axis=0, keepdims=True)  # [T, N, 3]
    # Scalar displacement magnitude per residue: [T, N]
    d = np.sqrt(np.sum(delta ** 2, axis=-1))

    tau_c = np.zeros(N)
    for i in range(N):
        x = d[:, i]
        var = np.var(x)
        if var < 1e-10:
            tau_c[i] = 0.0
            continue

        # FFT-based normalized autocorrelation
        x_centered = x - x.mean()
        n_fft = 2 * T
        X = np.fft.rfft(x_centered, n=n_fft)
        acf_full = np.fft.irfft(X * np.conj(X))[:T].real
        acf = acf_full[:max_lag] / acf_full[0]  # normalize so C(0) = 1

        # Fit exponential decay C(tau) = exp(-tau / tau_c)
        try:
            popt, _ = curve_fit(
                lambda t, tc: np.exp(-t / tc),
                lags, acf,
                p0=[10.0],
                bounds=(0.1, T),
                maxfev=2000,
            )
            tau_c[i] = popt[0]
        except Exception:
            # Fallback: integrated autocorrelation time (sum while positive)
            tau_c[i] = float(np.sum(acf[acf > 0]))

    return tau_c


def gogogo_autocorr(args):
    """
    Correlation between D-PLM per-residue embedding norms and per-residue
    autocorrelation times (tau_c) derived from MD trajectories.

    tau_c is sensitive to temporal frame ordering, unlike RMSF/DCCM, making
    this function suitable for the autocorrelation ablation test.

    Returns
    -------
    corr_list : list of float
        Per-protein Spearman correlations between embedding norms and tau_c.
    (rep_norm_list, autocorr_tau_list) : tuple of lists
        Aggregated residue norms and tau_c values across all proteins.
    """
    args.model, args.alphabet = load_model(args)

    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    analysis_base = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/'
    processed_list = []
    rep_norm_list = []
    autocorr_tau_list = []
    corr_list = []

    for file_name in os.listdir(datapath):
        print(f"processing...{file_name}")
        pid = "_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue

        pdb_file = os.path.join(analysis_base, f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))

        # Embeddings
        seq_emb = main(args)  # [seq_l+2, 1280]
        seq_emb = seq_emb[1:-1]
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)  # [N]

        # Find XTC trajectory (take first match, typically R1)
        xtc_pattern = os.path.join(analysis_base, f"{pid}_analysis", "*_R1.xtc")
        xtc_files = sorted(glob.glob(xtc_pattern))
        if not xtc_files:
            print(f"  No XTC found for {pid}, skipping.")
            processed_list.append(pid)
            continue
        xtc_file = xtc_files[0]

        # Load trajectory and collect C-alpha positions
        u = mda.Universe(pdb_file, xtc_file)
        ca = u.select_atoms("name CA")
        positions = np.array([ca.positions.copy() for _ in u.trajectory])  # [T, N_ca, 3]

        # Compute per-residue autocorrelation times
        tau_c = compute_autocorr_times(positions)  # [N_ca]

        # Align lengths in case of chain breaks or missing residues
        min_len = min(len(residue_norms), len(tau_c))
        residue_norms = residue_norms[:min_len]
        tau_c = tau_c[:min_len]

        rep_norm_list.extend(residue_norms)
        autocorr_tau_list.extend(tau_c)
        corr, _ = spearmanr(residue_norms, tau_c)
        corr_list.append(corr)
        processed_list.append(pid)

    return corr_list, (rep_norm_list, autocorr_tau_list)


from transformers import AutoTokenizer
import sys
from Bio.PDB import PDBParser, PPBuilder
sys.path.insert(0, "/work/nvme/bcnr/jyx/dplm/SeqDance-main/SeqDance-main/model/")
from model import ESMwrap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm2_select = 'model_35M'
model_select = 'seqdance' # 'seqdance' or 'esmdance'
dance_model = ESMwrap(esm2_select, model_select)
# Load the SeqDance model from huggingface
dance_model = dance_model.from_pretrained("ChaoHou/ESMDance")
dance_model = dance_model.to(device)
dance_model.eval()

def gogogo_seqdance():
    #
    #
    datapath = '/work/nvme/bcnr/jyx/dplm/Atlas_geom2vec_test/'
    # datapath_MDfeature = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test_R1/'
    processed_list=[]
    # rep_norm_list=[]
    # rmsf_list=[]
    # rep_dis_list=[]
    # dccm_list=[]
    corr_list=[]
    sig_list=[]
    for file_name in os.listdir(datapath):
        print(f"processing...{file_name}")
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/work/nvme/bcnr/jyx/dplm/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        sequence = str(pdb2seq(pdb_file))
        #
        # seq_emb = main(args) #[seq_l+2, 1280]
        raw_input = tokenizer([sequence], return_tensors="pt")
        raw_input = raw_input.to(device)
        with torch.no_grad():
            seq_emb = dance_model(raw_input, return_res_emb=True, return_attention_map=False, return_res_pred=False, return_pair_pred=False)
        seq_emb = seq_emb['res_emb'][:,1:-1,:].squeeze(0)
        # print(seq_emb.shape)
        distance_matrix = pairwise_distances(seq_emb.cpu(), metric="euclidean")  # or cosine
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        #
        rmsf_file = os.path.join("/work/nvme/bcnr/jyx/dplm/analysis/", f"{pid}_analysis", f"{pid}_RMSF.tsv")
        df = pd.read_csv(rmsf_file, sep="\t")
        r1 = df["RMSF_R1"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r1)
        # corr_cca = compute_cca_correlation(seq_emb.cpu().numpy(), r1)
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=1)
        # emb_proj = pca.fit_transform(seq_emb.cpu().numpy()).flatten()
        # corr_pca, p_pca = spearmanr(emb_proj, r1)
        corr, _ = spearmanr(residue_norms, r1)
        corr_list.append(corr)
        # sig_list.append(p_pca)
        # r2 = df["RMSF_R2"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r2)
        # r3 = df["RMSF_R3"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r3)
        # print(r1.shape)
        # MDfeature_file_r1 = os.path.join(datapath_MDfeature, f"{file_name[:-3]}.h5")
        # if os.path.exists(MDfeature_file_r1):
        #     rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r1)
        #     r1 = flatten_upper(distance_matrix)
        #     r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
        #     rep_dis_list.extend(r1)
        #     dccm_list.extend(r2)
        #
        # MDfeature_file_r2 = os.path.join(datapath_MDfeature, f"{file_name}_prod_R2_fit.h5")
        # if os.path.exists(MDfeature_file_r2):
        #     rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r2)
        #     r1 = flatten_upper(distance_matrix)
        #     r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
        #     rep_dis_list.extend(r1)
        #     dccm_list.extend(r2)
        
        # MDfeature_file_r3 = os.path.join(datapath_MDfeature, f"{file_name}_prod_R3_fit.h5")
        # if os.path.exists(MDfeature_file_r3):
        #     rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r3)
        #     r1 = flatten_upper(distance_matrix)
        #     r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
        #     rep_dis_list.extend(r1)
        #     dccm_list.extend(r2)
        #
        # corr, _ = spearmanr(r1, r2)
        # corr_rmsf, _ = spearmanr(residue_norms, rmsf)
        processed_list.append(pid)
    #
    return corr_list
	
	
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
datapath = '/work/nvme/bcnr/jyx/dplm/Atlas_geom2vec_test/'
# datapath_MDfeature = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test_R1/'
processed_list=[]
corr_list=[]

for file_name in os.listdir(datapath):
    print(f"processing...{file_name}")
    pid="_".join(file_name.split('_')[:2])
    if pid in processed_list:
        continue
    pdb_file = os.path.join("/work/nvme/bcnr/jyx/dplm/analysis/", f"{pid}_analysis", f"{pid}.pdb")
    sequences = [str(pdb2seq(pdb_file))]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    # The direction of the translation is indicated by two special tokens:
    # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
    # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
    sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                          for s in sequence_examples
                        ]
    
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples,
                                      add_special_tokens=True,
                                      padding="longest",
                                      return_tensors='pt').to(device)
    
    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(
                  ids.input_ids, 
                  attention_mask=ids.attention_mask
                  )
        
    seq_emb = embedding_repr.last_hidden_state[0, 1:-1,:]
    distance_matrix = pairwise_distances(seq_emb.cpu(), metric="euclidean")  # or cosine
    residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
    #
    rmsf_file = os.path.join("/work/nvme/bcnr/jyx/dplm/analysis/", f"{pid}_analysis", f"{pid}_RMSF.tsv")
    df = pd.read_csv(rmsf_file, sep="\t")
    r1 = df["RMSF_R1"].values
    #rep_norm_list.extend(residue_norms)
    #rmsf_list.extend(r1)
    corr, _ = spearmanr(residue_norms, r1)
    corr_list.append(corr)
    processed_list.append(pid)
	
	
import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your real lists)
method1 = [1.2, 1.5, 1.7, 2.0, 1.9]
method2 = [0.8, 1.0, 1.1, 1.3, 1.2]
method3 = [2.2, 2.4, 2.1, 2.8, 2.5]
method4 = [1.0, 0.9, 1.4, 1.3, 1.1]

data = [method1, method2, method3, method4]
labels = ["Method1", "Method2", "Method3", "Method4"]

plt.figure(figsize=(8, 5))

# Boxplot
plt.boxplot(data, labels=labels, showmeans=True)

# Overlay scatter points with jitter
for i, values in enumerate(data, start=1):
    x = np.random.normal(i, 0.04, size=len(values))  # jitter
    plt.scatter(x, values, alpha=0.7, s=30)

plt.ylabel("Value")
plt.title("Distribution Comparison with Individual Data Points")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.show()

def B_factor(args):
    #
    args.model,args.alphabet = load_model(args)
    #
    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    # datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
    processed_list=[]
    rep_norm_list=[]
    bf_list=[]
    for file_name in os.listdir(datapath):
        print(f"processing...{file_name}")
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))
        #
        seq_emb = main(args) #[seq_l+2, 1280]
        seq_emb = seq_emb[1:-1]
        # print(seq_emb.shape)
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        #
        bf_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_Bfactor.tsv")
        df = pd.read_csv(bf_file, sep="\t")
        r1 = df["Bfactor"].values.astype(np.float32)
        # rep_norm_list.extend(residue_norms)
        # bf_list.extend(r1)
        valid_mask = ~np.isnan(r1)
        # Apply mask to both arrays
        filtered_norms = residue_norms[valid_mask]
        filtered_bf = r1[valid_mask]
        rep_norm_list.extend(filtered_norms)
        bf_list.extend(filtered_bf)
        # r2 = df["RMSF_R2"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r2)
        # r3 = df["RMSF_R3"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r3)
        # print(r1.shape)
    
    return (rep_norm_list, bf_list)

def plddt(args):
    #
    args.model,args.alphabet = load_model(args)
    #
    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    # datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
    processed_list=[]
    rep_norm_list=[]
    plddt_list=[]
    for file_name in os.listdir(datapath):
        print(f"processing...{file_name}")
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))
        #
        seq_emb = main(args) #[seq_l+2, 1280]
        seq_emb = seq_emb[1:-1]
        # print(seq_emb.shape)
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        #
        plddt_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_pLDDT.tsv")
        df = pd.read_csv(plddt_file, sep="\t")
        r1 = df["pLDDT"].values.astype(np.float32)
        # rep_norm_list.extend(residue_norms)
        # bf_list.extend(r1)
        valid_mask = ~np.isnan(r1)
        # Apply mask to both arrays
        filtered_norms = residue_norms[valid_mask]
        filtered_plddt = r1[valid_mask]
        rep_norm_list.extend(filtered_norms)
        plddt_list.extend(filtered_plddt)
        # r2 = df["RMSF_R2"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r2)
        # r3 = df["RMSF_R3"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r3)
        # print(r1.shape)
    
    return (rep_norm_list, plddt_list)

def elbow2n(emb):
    # Elbow method to determine optimal number of clusters
    wcss = []
    K_range = range(1, 50)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(emb)
        wcss.append(kmeans.inertia_)  # Within-cluster sum of squares
    
    # Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, marker='o')
    plt.title('Elbow Method for RMSD-Based Clustering')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(K_range)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./1111.png')
    return 0

def rmsd(args):
    #
    args.model,args.alphabet = load_model(args)
    #
    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    processed_list=[]
    rep_norm_list=[]
    rmsd_list=[]
    for file_name in os.listdir(datapath):
        print(f"processing...{file_name}")
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))
        #
        seq_emb = main(args) #[seq_l+2, 1280]
        # seq_emb = seq_emb[0].cpu()
        seq_emb = np.linalg.norm(seq_emb[1:-1].cpu(), axis=0) #[1280]
        # seq_emb = np.sum(seq_emb[1:-1].cpu().numpy(), axis=0)/len(args.sequence) #[1280]
        # print(seq_emb.shape)
        # residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1) #[seq_l]
        #
        rmsd_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_RMSD.tsv")
        df = pd.read_csv(rmsd_file, sep="\t")
        r1 = df["RMSD_R1"].values
        rep_norm_list.append(seq_emb)
        rmsd_list.append(r1)
        # r2 = df["RMSF_R2"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r2)
        # r3 = df["RMSF_R3"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r3)
        # print(r1.shape)
    
    # return (rep_norm_list, bf_list)
    # Step 1: Load your data
    # rmsd_vectors: shape [N_proteins, T]
    # static_embeddings: shape [N_proteins, D]
    # rmsd_vectors = np.load("rmsd_vectors.npy")
    rmsd_vectors = np.array(rmsd_list)
    static_embeddings = np.array(rep_norm_list)
    # Optional: Normalize the RMSD vectors
    # scaler = StandardScaler()
    # rmsd_vectors = scaler.fit_transform(rmsd_vectors)
    
    # Step 2: Clustering
    # n_clusters = elbow2n(rmsd_vectors)
    n_clusters = 17  # set this based on elbow method or prior knowledge
    
    
    kmeans_rmsd = KMeans(n_clusters=n_clusters, random_state=0).fit(rmsd_vectors)
    kmeans_embed = KMeans(n_clusters=n_clusters, random_state=0).fit(static_embeddings)
    
    cluster_rmsd = kmeans_rmsd.labels_
    cluster_embed = kmeans_embed.labels_
    
    # Step 3: Clustering comparison metrics
    ari = adjusted_rand_score(cluster_rmsd, cluster_embed)
    nmi = normalized_mutual_info_score(cluster_rmsd, cluster_embed)
    
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    
    # Step 4: Visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embed = tsne.fit_transform(static_embeddings)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Clustered by RMSD")
    plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_rmsd, cmap='tab10')
    plt.subplot(1, 2, 2)
    plt.title("Clustered by Representation")
    plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_embed, cmap='tab10')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./rmsd.png')

def gyr(args):
    #
    args.model,args.alphabet = load_model(args)
    #
    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    processed_list=[]
    rep_norm_list=[]
    gyr_list=[]
    for file_name in os.listdir(datapath):
        print(f"processing...{file_name}")
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))
        #
        seq_emb = main(args) #[seq_l+2, 1280]
        # seq_emb = seq_emb[0].cpu()
        seq_emb = np.linalg.norm(seq_emb[1:-1].cpu(), axis=0) #[1280]
        # seq_emb = np.sum(seq_emb[1:-1].cpu().numpy(), axis=0)/len(args.sequence) #[1280]
        # print(seq_emb.shape)
        # residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1) #[seq_l]
        #
        gyr_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_gyrate.tsv")
        df = pd.read_csv(gyr_file, sep="\t")
        r1 = df["gyr_R1"].values
        rep_norm_list.append(seq_emb)
        gyr_list.append(r1)
        # r2 = df["RMSF_R2"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r2)
        # r3 = df["RMSF_R3"].values
        # rep_norm_list.extend(residue_norms)
        # rmsf_list.extend(r3)
        # print(r1.shape)
    
    # return (rep_norm_list, bf_list)
    # Step 1: Load your data
    # rmsd_vectors: shape [N_proteins, T]
    # static_embeddings: shape [N_proteins, D]
    # rmsd_vectors = np.load("rmsd_vectors.npy")
    gyr_vectors = np.array(gyr_list)
    static_embeddings = np.array(rep_norm_list)
    # Optional: Normalize the RMSD vectors
    # scaler = StandardScaler()
    # gyr_vectors = scaler.fit_transform(gyr_vectors)
    
    # Step 2: Clustering
    # n_clusters = elbow2n(gyr_vectors)
    n_clusters = 17  # set this based on elbow method or prior knowledge
    
    
    kmeans_gyr = KMeans(n_clusters=n_clusters, random_state=0).fit(gyr_vectors)
    kmeans_embed = KMeans(n_clusters=n_clusters, random_state=0).fit(static_embeddings)
    
    cluster_gyr = kmeans_gyr.labels_
    cluster_embed = kmeans_embed.labels_
    
    # Step 3: Clustering comparison metrics
    ari = adjusted_rand_score(cluster_gyr, cluster_embed)
    nmi = normalized_mutual_info_score(cluster_gyr, cluster_embed)
    
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    
    # Step 4: Visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embed = tsne.fit_transform(static_embeddings)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Clustered by gyr")
    plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_gyr, cmap='tab10')
    plt.subplot(1, 2, 2)
    plt.title("Clustered by Representation")
    plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_embed, cmap='tab10')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./gyr.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--model-location", type=str, help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth')
    parser.add_argument("--config-path", type=str, default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_3/config_vivit3.yaml', help="xx")
    parser.add_argument("--model-type", default='d-plm', type=str, help="xx")
    parser.add_argument("--sequence", type=str, help="xx")
    parser.add_argument("--output_path", type=str, default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/evaluate/', help="xx")
    args = parser.parse_args()

    #RMSF
    (rep_norm_list, rmsf_list), (rep_dis_list, dccm_list) = gogogo(args)
    corr, _ = spearmanr(rep_norm_list, rmsf_list)
    plot_and_fit(rep_norm_list, rmsf_list, 'rep_norm', 'rmsf', corr, args.output_path, 'rep_norm_rmsf_e')
    # corr, _ = spearmanr(rep_dis_list, dccm_list)
    # plot_and_fit(rep_dis_list, dccm_list, 'rep_dis', 'dccm', corr, args.output_path, 'rep_dis_dccm_lora2')
    #Bfactor
    (rep_norm_list, bf_list) = B_factor(args)
    corr, _ = spearmanr(rep_norm_list, bf_list)
    plot_and_fit(rep_norm_list, bf_list, 'rep_norm', 'Bfactor', corr, args.output_path, 'rep_norm_Bfactor_e')
    #Bfactor
    (rep_norm_list, plddt_list) = plddt(args)
    corr, _ = spearmanr(rep_norm_list, plddt_list)
    plot_and_fit(rep_norm_list, plddt_list, 'rep_norm', 'plDDT', corr, args.output_path, 'rep_norm_plDDT_e')
    #rmsd
    rmsd(args)
    #gyr
    gyr(args)





    

    




