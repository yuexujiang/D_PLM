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
     
task="data5"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=5
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task="data6"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=6
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task="data7"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=7
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task="data8"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=8
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs

task="data9"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/'
num=9
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_test'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs


task="DCCM_GNN_bigger"
config_path='./configs_hell/gvp_v2/config_DCCM_GNN.yaml'
result_path='./results/DCCM_GNN_bigger/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh
     
     
########################     
     
task="data_0"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=0
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_1"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=1
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=2
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_3"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=3
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_4"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=4
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_5"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=5
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_6"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=6
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_7"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=7
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_8"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=8
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="data_9"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/'
num=9
split_num=10
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_data'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
#########################

task="test_0"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=0
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_v2add'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="test_1"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=1
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_v2add'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="test_2"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=2
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_v2add'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="test_3"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=3
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_v2add'
sbatch --export=folder=$folder,num=$num,outfolder=$outfolder,split_num=$split_num, \
     -J ${task} \
     data_process.pbs
     
task="test_4"
folder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/v2_newly_added/'
num=4
split_num=5
outfolder='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_v2add'
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

task="ESM1v_test_vivit_nomlm_3500"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm/checkpoints/checkpoint_0003500.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm/config.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs

task="ESM1v_test_vivit_mlm_3500"
model_location='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm_mlm/checkpoints/checkpoint_0003500.pth'
config_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/results/dplm_mlm/config_mlm.yaml'
scoring_strategy='wt-mt-RLA' #"mask-marginals
sbatch --export=model_location=$model_location,config_path=$config_path,scoring_strategy=$scoring_strategy, \
       -J ${task} \
       ./ESM_1v_data/mutation_effect_ESM1v.pbs
#####################################################
