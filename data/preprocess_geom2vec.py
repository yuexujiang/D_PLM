import os
from geom2vec_main.geom2vec import create_model
from geom2vec_main.geom2vec.data import extract_mdtraj_info_folder, infer_traj
from Bio.PDB import PDBParser, PPBuilder
import os
import h5py
import argparse
import torch

# === Input Paths ===
# topology_file = "/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/4eo1_A/4eo1_A.pdb"              # /cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/4eo1_A/4eo1_A.pdb
# trajectory_folder = "/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/4eo1_A/"    # /cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_test/4eo1_A/4eo1_A_prod_R1_fit.xtc
# saving_path = "/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/"     # /cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/

def geom2vec_func(topology_file, trajectory_folder, saving_path):
    # === Create Model ===
    rep_model = create_model(
        model_type='vis',
        checkpoint_path='/cluster/pixstor/xudong-lab/yuexu/D_PLM/geom2vec_main/checkpoints/visnet_l6_h128_rbf64_r75.pth',  # adjust if needed
        cutoff=7.5,
        hidden_channels=128,
        num_layers=6,
        num_rbf=64,
        device='cuda'
    )
    
    # === Extract Positions, Atomic Numbers, CG Mapping ===
    position_list, atomic_numbers, segment_counts, dcd_files, _ = extract_mdtraj_info_folder(
        folder=trajectory_folder,
        top_file=topology_file,
        stride=10,                         # no skipping; get every frame
        selection='protein',             # only protein atoms
        file_postfix='.xtc',             # change here if needed
        exclude_hydrogens=True
    )
    
    # === Ensure Output Directory Exists ===
    os.makedirs(saving_path, exist_ok=True)
    # dcd_files = [os.path.splitext(os.path.basename(f))[0] for f in dcd_files]
    # === Inference ===
    infer_traj(
        model=rep_model,
        hidden_channels=128,
        batch_size=20,
        data=position_list,
        atomic_numbers=atomic_numbers,
        cg_mapping=segment_counts,
        saving_path=saving_path,
        file_name_list=[os.path.splitext(os.path.basename(f))[0] for f in dcd_files],
        sum_or_mean='sum',
        device='cuda'
    )
    # return dcd_files



# file_name_list=[os.path.splitext(os.path.basename(f))[0] for f in dcd_files]
# for f in file_name_list:
#         saving_filepath = os.path.join(saving_path, f"{f}.pt")
#         traj_rep = torch.load(saving_filepath)  # [T, M, 4, D]
#         # Step 1: Aggregate over time (mean or sum)
#         traj_mean = traj_rep.mean(dim=0)  # → [M, 4, D]
#         # Step 2: Flatten last two dims
#         traj_flat = traj_mean.reshape(traj_mean.shape[0], -1)  # → [M, 4*D]

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

def process_samples(data_folder2search, sub_list, output_folder):
    # Specify the parent folder (folder A)
    folder_A = data_folder2search
    samples = []
    
    # Iterate through each subfolder under folder_A
    #data={}
    for subfolder in os.listdir(folder_A):
        subfolder_path = os.path.abspath(os.path.join(folder_A, subfolder))
        
        # Check if the path is a directory
        if os.path.isdir(subfolder_path) and subfolder_path in sub_list:
            traj_files=[]
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".xtc"):
                    file_xtc_path = os.path.join(subfolder_path, file_name)
                    traj_files.append(file_xtc_path)
                    # num_xtc += 1
                elif file_name.endswith(".pdb"):
                    file_pdb_path = os.path.join(subfolder_path, file_name)
                    pid = file_name.split(".")[0]
                    # num_pdb += 1
            print(pid)
            sequence = pdb2seq(file_pdb_path)
            if len(sequence)>512:
                continue

            # === Input Paths ===
            topology_file = file_pdb_path            
            trajectory_folder = subfolder_path
            saving_path = output_folder     # /cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_geom2vec_test/
            geom2vec_func(topology_file, trajectory_folder, saving_path)
            for file_xtc_path in traj_files:
                file_path_base = os.path.basename(file_xtc_path)
                file_name=os.path.splitext(file_path_base)[0]
                geom2vec_filepath = os.path.join(saving_path, f"{file_name}.pt")
                traj_rep = torch.load(geom2vec_filepath)  # [T, M, 4, D]
                # Step 1: Aggregate over time (mean or sum)
                traj_mean = traj_rep.mean(dim=0)  # → [M, 4, D]
                # Step 2: Flatten last two dims
                traj_flat = traj_mean.reshape(traj_mean.shape[0], -1)  # → [M, 4*D]
                traj_flat = traj_flat.cpu().numpy()
                outputfile = os.path.join(output_folder, os.path.splitext(file_path_base)[0] + f".h5")
                if os.path.exists(outputfile):
                    continue
                print(f'processing {outputfile}')
                #data[pid]={'ind2seq':{}, 'ind2map':{}}
                #for i in range(data_len):
                #     data[pid]['ind2map'][i]=[]
                os.makedirs(os.path.dirname(outputfile), exist_ok=True)
                write_h5_file(outputfile, traj_flat, pid, sequence)

def split_subfolders_into_n_lists(folder_path, n=6):
    # 获取folder_path下所有一级子文件夹（完整路径）
    all_subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, d))]
    # 按字母排序（可选）
    all_subfolders.sort()
    # 创建n个空列表
    lists = [[] for _ in range(n)]
    # 均匀分配
    for i, subfolder in enumerate(all_subfolders):
        lists[i % n].append(os.path.abspath(subfolder))
    
    return lists

def write_h5_file(file_path, geom2vec, pid, seq):
    with h5py.File(file_path, 'w') as f:
        # f.create_dataset('seq', data=seq)
        f.create_dataset('geom2vec', data=geom2vec)
        f.create_dataset('pid', data=pid)
        f.create_dataset('seq', data=str(seq).encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--data_folder2search", help="xx", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_data/')
    parser.add_argument("--list_num", type=int, default=0, help="xx")
    parser.add_argument("--output_folder", default='/cluster/pixstor/xudong-lab/yuexu/D_PLM/Atlas_MDfeature_data', help="xx")
    parser.add_argument("--split_num", type=int, default=6, help="xx")

    
    args_main = parser.parse_args()
    lists = split_subfolders_into_n_lists(args_main.data_folder2search, n=args_main.split_num)
    process_samples(args_main.data_folder2search, 
                     sub_list=lists[args_main.list_num], output_folder=args_main.output_folder)