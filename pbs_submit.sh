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


task="test"
config_path='./configs_hell/gvp_v2/config.yaml'
result_path='./results/test/'
sbatch --export=config_path=$config_path,result_path=$result_path, \
     -J ${task} \
     run_yjm85.sh