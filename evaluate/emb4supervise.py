import argparse

import torch

import esm
import esm_adapterH

import numpy as np
import yaml
from utils.utils import load_configs
from collections import OrderedDict



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
    if args.model_type=="d-plm":
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

def emb(args):
    batch_converter = args.alphabet.get_batch_converter()
    
    data = [
        ("protein1", args.sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        wt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    
    wt_representation = wt_representation.squeeze(0) #only one sequence a time
    return wt_representation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--model-location", type=str, help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_lora2/checkpoints/checkpoint_best_val_dms_corr.pth')
    parser.add_argument("--config-path", type=str, default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_lora2/config_lora2.yaml', help="xx")
    parser.add_argument("--model-type", default='d-plm', type=str, help="xx")
    parser.add_argument("--sequence", type=str, help="xx")
    parser.add_argument("--output_path", type=str, default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/evaluate/', help="xx")
    args = parser.parse_args()

    args.model,args.alphabet = load_model(args) #test using dplm model and esm2 model, respectively

    #load wt seq
    emb_wt = emb(args) #get wt seq representation

    #load mt seq
    emb_mt = emb(args) #get mt seq representation

    # now both emb_wt and emb_mt should be tensor of shape [seq_Length, dim]
    # Suppose the muation index is ind, you can try extract the feature of this mutation using:
    #     option 1: emb_mt[ind]-emb_wt[ind]
    #     option 2: np.linalg.norm((emb_mt-emb_wt).to('cpu').detach().numpy(),axis=0)
    # both options should give a output tensor of shape [1280], use this emb and the label to train your mlp classifier.




