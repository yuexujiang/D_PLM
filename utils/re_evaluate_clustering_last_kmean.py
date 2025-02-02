import glob
import re
import torch
import argparse
#parser = argparse.ArgumentParser(description='evaluate_clustering')
import yaml
from utils import load_configs
from util_CATH_withswin import evaluate_with_CATHmore as evaluate_with_CATHmore_swin
from evaluation import *
import os

def load_config(file_path):
    with open(file_path, 'r') as stream:
        for line in stream:
            if "num_end_adapter_layers" in line:
               return line.split("num_end_adapter_layers:")[1].strip()

modellist={
          #"v3_adapter_tune_esmembeddingtable_contactv2_fixswincorrect_33esm2":"adapterH",
          #"v3_loraesm_tune_esmembeddingtable_contactv2_fixswincorrect":"lora",
          #"v3_adapter_tune_esmembeddingtable_contactv3_fixswincorrect":"adapterH",
          #"adapterH_tuneall_16esm2_gitcosine100lr001_out256_pool2":"adapterH",
          #"v3_adapterH_tune_allembeddingtable_contactv2_tinypretrain_16esm2_run2gitcosine100lr001":"adapterH",
          #"adapterH_tuneall_18esm2_gitcosine100lr001_out256_pool2":"adapterH",
          #"adapterH_tuneall_16esm2_gitcosine100lr001_out512_pool2":"adapterH"
          #"esm2contrastV4_tunall_weight1111_20esm_adapterH":"adapterH", #poor
          #"esm2contrastV4_tunall_weight1000_16esm_adapterH":"adapterH", #poor
          #"esm2contrastV4_tunall_weight1100_clip16esm_adapterH":"adapterH", #poor
          #"esm2contrastV4_tunall_weight1100_clip16esm_adapterH_m2":"adapterH", #m2 has lower loss poor
          #"esm2contrastV4_tunall_weight1100_clip16esm_adapterH_m3":"adapterH", $poor
          #"esm2contrastV4_tunall_weight1100_clip20esm_adapterH_m2":"adapterH",
          #"esm2contrastV4_tunall_weight1100_clip20esm_adapterH_m3":"adapterH",
          }

modellist={
#1272395 new submit
#"testadapterh_16tuneall_sgdlr001rescale_esmpool2_on16esm2_gitcosine100lr001_pool2":"adapterH", #good
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run2":"adapterH", #restart with resume=True this is based on testadapterh_16fixesmtable_sgdlr001mixlenpad_esmpool2
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3":"adapterH",
#1272396
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3fixswin":"adapterH",
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3finetunelastESM":"adapterH",
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3finetunelastESMlr0001":"adapterH",
#"config_12esmtuneall_sgdlr001mixlenpad_esmpool2_run2_testshuff":"adapterH",
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3_temp01":"adapterH",
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool3":"adapterH", #need to use pool_mode = 3 can use pool_mode=2
#"config_14fixesmtable_sgdlr001mixlenpad_esmpool3":"adapterH",
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool3_run2lr1e-5":"adapterH", #1272368 run again correct
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run2MLM_alttunelmhead":"adapterH", #1272359 run again correct
#"config_16fixesm_sgdlr001mixlenpad_esmpool2_MLMcontrast_testnolmheadbz20":"adapterH", #1272417
#done! correct!
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run4lr0001",
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run5lr1e-6",
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run3lr0001",
#1272394 
#"config_16fixesm_sgdlr001mixlenpad_esmpool2_MLM":"adapterH",
#"config_16fixesm_sgdlr001mixlenpad_esmpool2_MLM_fixswin":"adapterH",
#"config_16fixesm_sgdlr001mixlenpad_esmpool2_MLMcontrast_restart2":"adapterH"
}

modellist = {
#1272537
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool3":"adapterH", #need to use pool_mode = 3 can use pool_mode=2
#"config_16fixesmtable_sgdlr001mixlenpad_esmpool3_run2":"adapterH",
#1272538
"config_16fixesmtable_sgdlr001mixlenpad_esmpool3_run2lr1e-5":"adapterH", #1272368 run again correct
}
pool_mode=3

if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

evaluate_last=10
for modelname in modellist.keys():
    esmType = modellist[modelname]
    file_list = glob.glob(modelname + "/checkpoints/checkpoint_*.pth")
    outputfile=open(modelname+"/evaluate_CATHmore_"+str(pool_mode)+".txt",'w')
    sorted_filelist = sorted(file_list, key=lambda x: int(x.split("/")[-1].split('_')[1].split('.')[0]))
    figure_output_folder="kmeans"+str(pool_mode)
    for checkpoint_path in sorted_filelist[-evaluate_last:]:
        if esmType=="lora":
            checkpoint_path=checkpoint_path+"_lora"
        
        print(checkpoint_path)
        #config = int(load_config(modelname+"/config.yml"))
        if os.path.exists(modelname+'/config.yml'):
           config_path=modelname+"/config.yml"
        else:
           config_path = modelname+"/config.yaml"
        
        with open(config_path) as file:
            configs_dict = yaml.full_load(file)
        
        config = load_configs(configs_dict)
        n_iter= int(re.findall(r'\d+', checkpoint_path)[-1])
        #"""
        existingmodel,batch_converter=init_model(esmType,
            checkpoint_path,
            esm2_pretrain="esm2_t33_650M_UR50D",
            existingmodel=None,num_end_adapter_layers=config.model.esm_encoder.adapter_h,
            maxlength=512,
            pool_mode=pool_mode,
            device=device
            )
        
        existingmodel.eval()
        cathpath="/mnt/pixstor/dbllab/duolin/CATH/seq_subfamily/Rep_subfamily_basedon_S40pdb.fa"
        scores_cath=evaluate_with_cath_more(device, modelname + "/" + figure_output_folder, steps=n_iter,
                                            esmType=esmType,
                                            batch_size=10,
                                            cathpath=cathpath,
                                            existingmodel=existingmodel
                                            )
        
        deaminasepath="/mnt/pixstor/dbllab/duolin/simCLR/Deaminases_clustering/Table_S1.xlsx"
        scores_deaminase=evaluate_with_deaminase(device,modelname+"/"+figure_output_folder,steps=n_iter,
                    esmType=esmType,
                    batch_size=8,
                    deaminasepath=deaminasepath,
                    existingmodel=existingmodel)
        
        kinasepath="/mnt/pixstor/dbllab/duolin/simCLR/kinase_clustering/GPS5.0_homo_hasPK_with_kinasedomain.txt"
        scores_kinase=evaluate_with_kinase(device,modelname+"/"+figure_output_folder,steps=n_iter,
                    esmType=esmType,
                    batch_size=8,
                    kinasepath=kinasepath,
                    existingmodel=existingmodel
                    )
        
        
        outputfile.write(f"step: {n_iter}\tdigit_num_1:{scores_cath[0]}({scores_cath[1]})\tdigit_num_2:{scores_cath[3]}({scores_cath[4]}\tdigit_num_3:{scores_cath[6]}({scores_cath[7]})\n")
        outputfile.write(f"step:{n_iter}\tdigit_num_1_ARI:{scores_cath[2]}\tdigit_num_2_ARI:{scores_cath[5]}\tdigit_num_3_ARI:{scores_cath[8]}\n")
        outputfile.write(f"step: {n_iter}\tdeaminase_score:{scores_deaminase[0]}({scores_deaminase[1]})\tARI:{scores_deaminase[2]}\n")
        outputfile.write(f"step: {n_iter}\tkinase_score:{scores_kinase[0]}({scores_kinase[1]})\tARI:{scores_kinase[2]}\n")
        #"""
        scores_cath=evaluate_with_CATHmore_swin(device,modelname+"/"+figure_output_folder,steps=n_iter,
                maxlength=512,
                batch_size=8,
                checkpoint_path=checkpoint_path,
                cathpath="/mnt/pixstor/dbllab/duolin/CATH/CATH_4_3_0_non-rep_superfamily_contactmap_nopad.npz",
                existingmodel=None,
                rescale=config.train_settings.rescale,
                pool_mode=pool_mode,
                )
        
        outputfile.write(f"step:{n_iter}\tswin_digit_num_1:{scores_cath[0]:.4f}({scores_cath[1]:.4f})\tswin_digit_num_2:{scores_cath[3]:.4f}({scores_cath[4]:.4f})\tswin_digit_num_3:{scores_cath[6]:.4f}({scores_cath[7]:.4f})\n")
        outputfile.write(f"step:{n_iter}\tswin_digit_num_1_ARI:{scores_cath[2]}\tswin_digit_num_2_ARI:{scores_cath[5]}\tswin_digit_num_3_ARI:{scores_cath[8]}\n")
    
    
    outputfile.close()

 