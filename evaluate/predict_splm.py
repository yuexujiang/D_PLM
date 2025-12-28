# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import string

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import esm
import esm_adapterH
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
import yaml
from utils.utils import load_configs,get_logging
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from peft import PeftModel, LoraConfig, get_peft_model

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
            # elif  any('lora' in name for name, _ in model.named_modules()):
            else:
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

"""
class DataSet(Dataset):
        def __init__(self, seqids,pad_seq):
            self.seqids = seqids
            self.pad_seq = pad_seq
        
        def __len__(self):
            return len(self.pad_seq)
        
        def __getitem__(self, index):
            return index, self.seqids[index], self.pad_seq[index]
"""



def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )
    # fmt: off
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        # nargs="+",
        default="/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm_mlm/checkpoints/checkpoint_0000450.pth"
    )
    parser.add_argument(
    "--config-path",
    type=str,
    default="/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm_mlm/config_mlm.yaml"
    )
    parser.add_argument(
    "--project-path",
    type=str,
    default="/cluster/pixstor/xudong-lab/yuexu/D_PLM/"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="s-plm",
        help="default s-plm, select from ESM2,ESM-1v",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-input",
        type=pathlib.Path,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=0,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="mask-marginals",
        choices=["wt-mt-whole","wt-mt-RLA","wt-mt-marginals","wt-marginals","mask-marginals"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA in a3m format (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=100,
        help="number of sequences to select from the start of the MSA"
    )
    parser.add_argument(
    '--ref-name',
    type=str,
    )
    parser.add_argument(
    '--similarity',
    type=str,
    default="euclidean_distance",
    choices=["cosine","euclidean_distance","correlation"],
    help=""
    )
    
    # fmt: on
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser



def gen_mt_seq(row,sequence,offset_idx):
   wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
   assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
   return sequence[:idx] + mt + sequence[(idx + 1) :]

def label_row(row, sequence, token_probs, alphabet, offset_idx):
    if ":" not in row:
       base = row[1:-1]
       if len(base)==0:
           return np.nan
       
       wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1] #offset-idx is 24 for this data row is mutant e.g. H24C
       if idx>=len(sequence):
           return np.nan
       
       assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
       if "_" in wt or "_" in mt:
            return np.nan
       
       wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
       # add 1 for BOS
       score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
       return score.item()
    else:
        score=0
        for row_i in row.split(":"):
            base = row[1:-1]
            if len(base)==0:
                return np.nan
            
            wt, idx, mt = row_i[0], int(row_i[1:-1]) - offset_idx, row_i[-1] #offset-idx is 24 for this data row is mutant e.g. H24C
            if idx>=len(sequence):
                return np.nan
            
            if "_" in wt or "_" in mt:
                return np.nan
            
            assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
            wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
            # add 1 for BOS
            score += token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        
        return score.item()/len(row.split(":"))



def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    
    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]
    
    # encode the sequence
    data = [
        ("protein1", sequence),
    ]
    
    batch_converter = alphabet.get_batch_converter()
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    
    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)



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


from pathlib import Path

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

def main(args):
    if args.scoring_strategy in ["wt-mt-whole","wt-mt-RLA","wt-mt-marginals"]:
       path=os.path.join(args.dms_output,args.scoring_strategy,args.similarity)
    else:
        path=os.path.join(args.dms_output,args.scoring_strategy)
    
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    # Load the deep mutational scan
    df = pd.read_csv(args.dms_input)
    tqdm.pandas()
    #df["mt-seq"] = df.progress_apply(lambda row: gen_mt_seq(row[args.mutation_col], args.sequence,args.offset_idx),axis=1)
    #dataset_val = DataSet(seqids,seqs)
    
    
    batch_converter = args.alphabet.get_batch_converter()
    
    data = [
        ("protein1", args.sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    if args.scoring_strategy in ["wt-mt-whole","wt-mt-RLA","wt-mt-marginals"]: #default
        if args.scoring_strategy=="wt-mt-whole":
           mode = "whole"
        elif args.scoring_strategy=="wt-mt-RLA":
           mode = "RLA"
        elif args.scoring_strategy == "wt-mt-marginals":
           mode = "marginals"
        tqdm.pandas()
        with torch.no_grad():
            wt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
        
        wt_representation = wt_representation.squeeze(0) #only one sequence a time
        df[args.modelname] = df.progress_apply(
            lambda row: compute_repdiff(
                row[args.mutation_col],
                args.sequence,
                args.model,
                args.alphabet,
                args.offset_idx,
                wt_representation,
                mode=mode,
                similarity=args.similarity
            ),
            axis=1,
        )
    elif args.scoring_strategy == "wt-marginals":
        with torch.no_grad():
            token_probs = torch.log_softmax(args.model(batch_tokens.cuda())["logits"], dim=-1) #1,256,33
            #token_probs = torch.log_softmax(args.model.model_seq.esm2(batch_tokens.cuda())["logits"], dim=-1)
            df[args.modelname] = df.apply(
                lambda row: label_row(
                    row[args.mutation_col],
                    args.sequence,
                    token_probs,
                    args.alphabet,
                    args.offset_idx,
                ),
                axis=1,
            )
    elif args.scoring_strategy == "mask-marginals":
         all_token_probs = []
         for i in tqdm(range(batch_tokens.size(1))):#265 = 263+2
             batch_tokens_masked = batch_tokens.clone()        
             batch_tokens_masked[0, i] = args.alphabet.mask_idx        
             with torch.no_grad():        
                 token_probs = torch.log_softmax(        
                     args.model(batch_tokens_masked.cuda())["logits"], dim=-1        
                 )
             
             all_token_probs.append(token_probs[:, i])  # vocab size        
         
         token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)        
         df[args.modelname] = df.apply(        
             lambda row: label_row(        
                 row[args.mutation_col],        
                 args.sequence,        
                 token_probs,        
                 args.alphabet,        
                 int(args.offset_idx),        
             ),        
             axis=1,
         )
    elif args.scoring_strategy == "pseudo-ppl":
            tqdm.pandas()
            df[args.modelname] = df.progress_apply(
                lambda row: compute_pppl(
                    row[args.mutation_col], args.sequence, args.model, args.alphabet, args.offset_idx
                ),
                axis=1,
            )
    
    
    ## Calculate statistics from results
    esm_rla_corr = df[args.ref_name].corr(df[args.modelname],method = 'pearson')
    esm_rla_spearmn = df[args.ref_name].corr(df[args.modelname],method='spearman')    
    fig = plt.figure()
    plt.scatter(df[args.ref_name], df[args.modelname], s=1)
    plt.xlabel('Observed')
    plt.ylabel(f'Predicted ({args.modelname} scores)')
    
    plt.title(f'PCC: {round(esm_rla_corr, 3)} spearmanr: {round(esm_rla_spearmn,3)}')
    plt.savefig(os.path.join(path,args.filekey+'_pcc.png'))
    
    df.to_csv(os.path.join(path,args.filekey+"_predict.csv"))
    if args.scoring_strategy in ["wt-mt-whole","wt-mt-RLA","wt-mt-marginals"]:
       args.logging.info(f'{args.model_type},{args.scoring_strategy},{args.similarity},{args.filekey}')
       args.logging.info(f'PCC: {round(esm_rla_corr, 3)} spearmanr: {round(esm_rla_spearmn,3)}')
    else:
       args.logging.info(f'{args.model_type},{args.scoring_strategy},{args.filekey}')
       args.logging.info(f'PCC: {round(esm_rla_corr, 3)} spearmanr: {round(esm_rla_spearmn,3)}')
    
    return esm_rla_spearmn


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    #for debug
    """
    args.config_path = "/data/duolin/splm_gvpgit/results/checkpoints_configs/config_plddtallweight_noseq_rotary_foldseek.yaml"
    #args.model_location="/data/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth"
    #args.model_location="/data/duolin/splm/results/checkpoints_configs/checkpoint_0280000.pth" wait!
    args.model_location="/data/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_run2_0020500.pth"
    args.dms_input = "/data/duolin/splm_gvpgit/splm/esm-main/examples/variant-prediction/data/BLAT_ECOLX_Ranganathan2015_labeled.csv"
    args.sequence = "HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    args.mutation_col = "mutant"
    args.offset_idx = 24
    args.ref_name = '2500'
    args.scoring_strategy = "mask-marginals"#"wt-marginals"
    #args.scoring_strategy = "wt-marginals"
    #args.scoring_strategy = "wt-mt-whole"
    #args.scoring_strategy = "wt-mt-marginals"
    #args.scoring_strategy = "wt-mt-RLA"
    #args.scoring_strategy = "wt-mt-RLA"
    
    #args.similarity = "euclidean_distance" #mse euclidean_distance
    args.model_type = "s-plm"
    args.dms_output = "test_mutation_ESM1vdata/s-plm/20500/"
    
    #args.dms_output = "./test_mutation_ESM1vdata/ESM2"
    #args.model_type = "ESM2"
    """
    #debug end
    #result = main(args)
    
    #run for all 41 datasets
    #args.model_location = "/data/duolin/splm_gvpgit/results/"\
    #                      "checkpoints_configs/checkpoint_run2_0020500.pth"
    #args.model_location = "/data/duolin/splm_gvpgit/results/checkpoints_configs/checkpoint_0280000.pth" #12 layers
    #args.model_location = "/data/duolin/splm_gvpgit/results/"\
    #                      "checkpoints_configs/freezepretrain_MLM_separatetable_0043000_on_280000.pth"
    # args.model_location = "/data/duolin/splm_gvpgit/results/"\
    #                       "checkpoints_configs/freezepretrain_MLM_separatetable_0069000_on_280000.pth"
    #args.config_path = "/data/duolin/splm_gvpgit/results/"\
    #                   "checkpoints_configs/config_plddtallweight_noseq_rotary_foldseek.yaml"
    
    #args.scoring_strategy = "mask-marginals"
    #args.scoring_strategy = "wt-marginals"
    
    #args.scoring_strategy = "wt-mt-RLA"
    #args.similarity = "euclidean_distance"
    
    
    
    for args.model_type in ["s-plm",]: #["ESM2","s-plm"]:
      if args.model_type == "ESM2":
           args.modelname = "ESM2-650M"
      elif args.model_type=="s-plm":
           args.modelname = "/".join(args.model_location.split("/")[-1:])#"s-plm-seq"
      elif args.model_type == "ESM-1v":
           args.modelname = "ESM-1v"
      
      args.dms_output = args.project_path+"ESM_1v_data/results/"+args.modelname
      args.model,args.alphabet = load_model(args)
      pathlib.Path(args.dms_output).mkdir(parents=True, exist_ok=True)
      args.logging = get_logging(args.dms_output)
      
      outputfilename = os.path.join(args.project_path,"ESM_1v_data/results/",'ESM-1v_data_summary_start_end_seq_splm_esm2.csv')
      import shutil
      
      if os.path.exists(outputfilename):
          methodname=args.model_location.split("/")[-3]
          newname = os.path.join(args.project_path,"ESM_1v_data/results/", f"{methodname}.csv")
          shutil.copy(outputfilename, newname)
          outputfilename = newname
          df=pd.read_csv(outputfilename)
      else:
          methodname=args.model_location.split("/")[-3]
          newname = os.path.join(args.project_path,"ESM_1v_data/data_wedownloaded/41 mutation dataset/", f"{methodname}.csv")
          shutil.copy(args.project_path+"ESM_1v_data/data_wedownloaded/41 mutation dataset/ESM-1v_data_summary_start_end_seq.csv", newname)
          outputfilename = newname
          df=pd.read_csv(outputfilename)
        #   df = pd.read_csv("/cluster/pixstor/xudong-lab/yuexu/D_PLM/ESM_1v_data/data_wedownloaded/41 mutation dataset/ESM-1v_data_summary_start_end_seq.csv")
      
      for index, row in df.iterrows():
      #for index, row in df.iloc[index:].iterrows():
        wt_seq = row['WT seq']
        if wt_seq is np.nan:
            continue
        
        args.filekey = row['Dataset_file']
        ### just for mutation effect to stability
        print(args.filekey)
        if not args.filekey == 'TPMT_HUMAN_Fowler2018' and not args.filekey == 'PTEN_HUMAN_Fowler2018' \
            and not args.filekey == 'BG_STRSQ_hmmerbit' \
                and not args.filekey == 'DLG4_RAT_Ranganathan2012' \
                    and not args.filekey == 'RL401_YEAST_Bolon2014' \
                        and not args.filekey == 'UBE4B_MOUSE_Klevit2013-singles' \
                            and not args.filekey == 'YAP1_HUMAN_Fields2012-singles' \
                                and not args.filekey == 'AMIE_PSEAE_Whitehead' \
                                    and not args.filekey == 'RASH_HUMAN_Kuriyan':
            # if args.scoring_strategy in ["wt-mt-whole","wt-mt-RLA","wt-mt-marginals"]:
            #    df.loc[index,args.modelname+"-"+args.scoring_strategy+"-"+args.similarity] = 0
            # else:
            #     df.loc[index,args.modelname+"-"+args.scoring_strategy] = 0
            continue
        path_test = os.path.join(args.project_path,"ESM_1v_data/data_wedownloaded/41 mutation dataset/ESM-1v-41/test",row['Dataset_file']+".csv")
        path_val = os.path.join(args.project_path,"ESM_1v_data/data_wedownloaded/41 mutation dataset/ESM-1v-41/validation",row['Dataset_file']+".csv")
        if os.path.exists(path_test):
           args.dms_input = path_test
        else:
           args.dms_input = path_val
        
        args.sequence = str(wt_seq)
        args.offset_idx = int(row['offset_idx'])
        args.ref_name = row['Name(s) in Reference']
        result = main(args)
        if args.scoring_strategy in ["wt-mt-whole","wt-mt-RLA","wt-mt-marginals"]:
           df.loc[index,args.modelname+"-"+args.scoring_strategy+"-"+args.similarity] = result
        else:
            df.loc[index,args.modelname+"-"+args.scoring_strategy] = result
    
    
    
    df.to_csv(outputfilename,index=False)
    



"""
x1 = df["esm1v_t33_650M_UR90_1"].tolist()
x2 = df["esm1v_t33_650M_UR90_2"].tolist()
x3 = df["esm1v_t33_650M_UR90_3"].tolist()
x4 = df["esm1v_t33_650M_UR90_4"].tolist()
x5 = df["esm1v_t33_650M_UR90_5"].tolist()

y = df["2500"].tolist() #4996
x = np.mean([x1,x2,x3,x4,x5],0)
corr, p_value = spearmanr(x, y)
"""


#0.508 spearmanr vs 0.722 ESM2
#pcc 0.493 vs 0.699 ESM2
#mask-marginals PCC: 0.532 spearmanr: 0.543



