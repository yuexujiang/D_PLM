test my change
1 Installation by install_gpu
2 Train by run.sh
3 Key configs:
  
3.1 if train from scratch, set resume.resume to False
  and run: 
 python train.py --config_path configs.yaml --result_path ./results/model_keyword
  
 If resume from pretrain, set resume.resume to True
  and run:
  
python train.py --config_path configs.yaml \
--result_path ./results/model_keyword/ \
--resume_path "./results/model_keyword/checkpoints/checkpoint_0100000.pth"

3.2 train_settings.plddt_weight whether to use plddt as sample weight, only for residue level loss, if True, use plddt for the correspoding positive residue as the sample weight

3.3 batch_size = 20 is for 80G GPUs

3.4 residue_batch_size: how many residue pairs selected for one batch. The toltal number will be batch_size x (protein length)

3.5 Parameters for projectors (protein or residue)
  residue_out_dim: 128
  protein_out_dim: 256
  residue_num_projector: 1
  protein_num_projector: 2
  residue_inner_dim: 1024
  protein_inner_dim: 4096

3.6 How many adapter layers
model.esm_encoder.adapter_h.num_end_adapter_layers  more adapter layers, better results for CATH but worse results for sequence features. For gvp model, we can use a small value.



4 for evaluation, I called utils_evaluation to evaluate sequence-based ESM2 model, and utils_CATH_withstruct to evaluate structure-based gvp model just for CATH case.
  After the training, scripts, like plot_loss.py, plot_cluster_results_.py, plot_ARI_results_hellbender.py can be used to evaluate the performance of protein clustering tasks.

5 how to do data transfer? 