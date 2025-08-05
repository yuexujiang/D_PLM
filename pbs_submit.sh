task="data0"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=0
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs


task="data1"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=1
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs
     
     
task="data2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=2
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs
     
     
task="data3"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=3
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs
     
     
task="data4"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=4
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs
     

task="data5"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs
     
     

task="test"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=0
split_num=1
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs


task="dplm"
config_path='./configs_hell/gvp_v2/config.yaml'
result_path='./results/dplm/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh
     
task="resume_test2"
config_path='./configs_hell/gvp_v2/config.yaml'
result_path='./results/test2/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh
     
task="mlm"
config_path='./configs_hell/gvp_v2/config_mlm.yaml'
result_path='./results/dplm_mlm/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh
     
task="mlm_small"
config_path='./configs_hell/gvp_v2/config_mlm_small.yaml'
result_path='./results/dplm_mlm_small/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh
     


     
task="data0"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=0
split_num=6
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data1"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=1
split_num=6
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=2
split_num=6
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data3"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=3
split_num=6
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data4"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=4
split_num=6
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data5"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=5
split_num=6
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task="data0"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=0
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data1"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=1
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=2
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data3"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=3
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data4"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=4
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     

########################     
     
task="data_0"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=0
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_1"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=1
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=2
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_3"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=3
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_4"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=4
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     

###########################

task='v2_0'
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=0
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_v2added'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task='v2_1'
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=1
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_v2added'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task='v2_2'
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=2
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_v2added'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task='v2_3'
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=3
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_v2added'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task='v2_4'
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=4
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/processed_Atlas_v2added'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
#########################

task="test_0"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=0
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="test_1"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=1
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task="test_2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=2
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task="test_3"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=3
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs 

task="test_4"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=4
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_vivit_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

     
#####################
task="geom2vec_tematt"
config_path='./configs_hell/gvp_v2/config_geom2vec_tematt.yaml'
result_path='./results/geom2vec_tematt_v2add/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

#####################
task="geom2vec_tematt_morestep"
config_path='./configs_hell/gvp_v2/config_geom2vec_tematt_morestep.yaml'
result_path='./results/geom2vec_tematt_morestep/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="geom2vec_tematt_morestep2"
config_path='./configs_hell/gvp_v2/config_geom2vec_tematt_morestep_2.yaml'
result_path='./results/geom2vec_tematt_morestep2/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="geom2vec_tematt_morestep3"
config_path='./configs_hell/gvp_v2/config_geom2vec_tematt_morestep_3.yaml'
result_path='./results/geom2vec_tematt_morestep3/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="geom2vec_temcnn"
config_path='./configs_hell/gvp_v2/config_geom2vec_temcnn.yaml'
result_path='./results/geom2vec_temcnn/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="geom2vec_temcnn_fix"
config_path='./configs_hell/gvp_v2/config_geom2vec_temcnn_fix.yaml'
result_path='./results/geom2vec_temcnn_fix/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="geom2vec_temcnn_fix_mlm"
config_path='./configs_hell/gvp_v2/config_geom2vec_temcnn_fix_mlm.yaml'
result_path='./results/geom2vec_temcnn_fix_mlm/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="geom2vec_temlstmatt"
config_path='./configs_hell/gvp_v2/config_geom2vec_temlstmatt.yaml'
result_path='./results/geom2vec_temlstmatt/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="geom2vec_vivit"
config_path='./configs_hell/gvp_v2/config_vivit_resize.yaml'
result_path='./results/vivit_resize/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="vivit_meanrep"
config_path='./configs_hell/gvp_v2/config_vivit_meanrep.yaml'
result_path='./results/vivit_meanrep/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="vivit_splm"
config_path='./configs_hell/gvp_v2/config_vivit_splm.yaml'
result_path='./results/vivit_splm/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="vivit_hyper1"
config_path='./configs_hell/gvp_v2/config_vivit_hyper1.yaml'
result_path='./results/vivit_hyper1/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="vivit_big"
config_path='./configs_hell/gvp_v2/config_vivit_resize_big.yaml'
result_path='./results/vivit_big/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="vivit_big2"
config_path='./configs_hell/gvp_v2/config_vivit_resize_big2.yaml'
result_path='./results/vivit_big2/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="vivit_resize"
config_path='./configs_hell/gvp_v2/config_vivit_resize.yaml'
result_path='./results/vivit_resize2/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh

task="vivit"
config_path='./configs_hell/gvp_v2/config.yaml'
result_path='./results/vivit/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh
#####################
#need to submit in sbatch, need long time
task="ESM1v_test"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/DCCM_GNN_bigger/checkpoints/checkpoint_0001000.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/DCCM_GNN_bigger/config_DCCM_GNN.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs


task="ESM1v_test_geom2vec"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_v2add/checkpoints/checkpoint_0002000.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_v2add/config_geom2vec_tematt.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="ESM1v_test_geom2vec_morestep"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_morestep/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_morestep/config_geom2vec_tematt_morestep.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="ESM1v_test_geom2vec_morestep2"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_morestep2/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_morestep2/config_geom2vec_tematt_morestep_2.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="ESM1v_test_geom2vec_morestep3"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_morestep3/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_tematt_morestep3/config_geom2vec_tematt_morestep_3.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="ESM1v_test_vivit_mlm_450"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm_mlm/checkpoints/checkpoint_0000450.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm_mlm/config_mlm.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="ESM1v_test_vivit_nomlm_750"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm/checkpoints/checkpoint_0000750.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm/config.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="ESM1v_test_geom2vec_temcnn"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temcnn/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temcnn/config_geom2vec_temcnn.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="geom2vec_temcnn_fix"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temcnn_fix/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temcnn_fix/config_geom2vec_temcnn_fix.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="geom2vec_temcnn_fix_mlm"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temcnn_fix_mlm/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temcnn_fix_mlm/config_geom2vec_temcnn_fix_mlm.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="geom2vec_temlstmatt"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temlstmatt/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/geom2vec_temlstmatt/config_geom2vec_temlstmatt.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="vivit_resize"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_resize/checkpoints/checkpoint_0004950.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_resize/config_vivit_resize.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="vivit_meanrep"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_meanrep/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_meanrep/config_vivit_meanrep.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs


task="vivit_splm"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_splm/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_splm/config_vivit_splm.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="vivit_hyper"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_hyper1/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_hyper1/config_vivit_hyper1.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="vivit_big2"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_big2/checkpoints/checkpoint_best_val_whole_loss.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/vivit_big2/config_vivit_resize_big2.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs
#####################################################
task="v2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_gvp_data/'
sbatch --export=folder=$folder,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs

task="data"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_gvp_data/'
sbatch --export=folder=$folder,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs

task="test"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_gvp_test/'
sbatch --export=folder=$folder,outfolder=$outfolder, \
     -J ${task} \
     data_process.pbs