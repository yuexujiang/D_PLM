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
import h5py
import pandas as pd


def load_checkpoints(model,checkpoint_path):
        if checkpoint_path is not None:
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
                
                model.load_state_dict(new_ordered_dict, strict=False)
            else:
                #this model does not contain esm2
                new_ordered_dict = OrderedDict()
                for key, value in checkpoint['state_dict1'].items():
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
        
        # inference for each model
        model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)
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
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--model-location", type=str, help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt/checkpoints/checkpoint_0002000.pth')
    parser.add_argument("--config-path", type=str, default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt/config_geom2vec_tematt.yaml', help="xx")
    parser.add_argument("--model-type", default='s-plm', type=str, help="xx")
    parser.add_argument("--sequence", type=str, help="xx")
    args = parser.parse_args()
    
    args.model,args.alphabet = load_model(args)

    datapath = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/'
    datapath_MDfeature = '/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test/'

    processed_list=[]
    rep_norm_list=[]
    rmsf_list=[]
    rep_dis_list=[]
    dccm_list=[]
    for file_name in os.listdir(datapath):
        pid="_".join(file_name.split('_')[:2])
        if pid in processed_list:
            continue
        pdb_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}.pdb")
        args.sequence = str(pdb2seq(pdb_file))
        
        seq_emb = main(args) #[seq_l+2, 1280]
        seq_emb = seq_emb[1:-1]
        print(seq_emb.shape)
        distance_matrix = pairwise_distances(seq_emb, metric="euclidean")  # or cosine
        residue_norms = np.linalg.norm(seq_emb, axis=1)

        rmsf_file = os.path.join("/cluster/pixstor/xudong-lab/yuexu/D_PLM/analysis/", f"{pid}_analysis", f"{pid}_RMSF.tsv")
        df = pd.read_csv(rmsf_file, sep="\t")
        r1 = df["RMSF_R1"].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r1)
        r2 = df["RMSF_R2"].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r2)
        r3 = df["RMSF_R3"].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r3)
        print(r1.shape)

        MDfeature_file_r1 = os.path.join(datapath_MDfeature, f"{file_name}_prod_R1_fit.h5")
        if os.path.exists(MDfeature_file_r1):
            rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r1)
            r1 = flatten_upper(distance_matrix)
            r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
            rep_dis_list.extend(r1)
            dccm_list.extend(r2)
        
        MDfeature_file_r2 = os.path.join(datapath_MDfeature, f"{file_name}_prod_R2_fit.h5")
        if os.path.exists(MDfeature_file_r2):
            rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r2)
            r1 = flatten_upper(distance_matrix)
            r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
            rep_dis_list.extend(r1)
            dccm_list.extend(r2)
        
        MDfeature_file_r3 = os.path.join(datapath_MDfeature, f"{file_name}_prod_R3_fit.h5")
        if os.path.exists(MDfeature_file_r3):
            rmsf, dccm, mi, seq, pid = load_h5_file_v2(MDfeature_file_r3)
            r1 = flatten_upper(distance_matrix)
            r2 = flatten_upper(1 - np.abs(dccm))  # higher = more correlated motion
            rep_dis_list.extend(r1)
            dccm_list.extend(r2)
        
        # corr, _ = spearmanr(r1, r2)
        # corr_rmsf, _ = spearmanr(residue_norms, rmsf)

        processed_list.append(pid)

    




