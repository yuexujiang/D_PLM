import math
import h5py
import psutil
import subprocess
import os


def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
    return seq, n_ca_c_o_coord, plddt_scores

#seq, n_ca_c_o_coord, plddt_scores,ss,sasa,node_degree,edge_index,edge_dis,edge_localorientation=load_h5_file_v2("/data/duolin/train_PLK1/datasets/structuresAs_hdf5_v2/AF-Q8WWH5-F1-model_v4_chain_id_A.h5")
def load_h5_file_v2(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
        ss = f['ss'][:]
        ss=''.join([s.decode('utf-8') for s in ss])
        sasa = f['sasa'][:]
        node_degree = f['node_degree'][:]
        edge_index = f['edge_index'][:]
        edge_dis  = f['edge_dis'][:]
        edge_localorientation = f['edge_localorientation'][:]
    
    return seq, n_ca_c_o_coord, plddt_scores,ss,sasa,node_degree,edge_index,edge_dis,edge_localorientation



def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)


def extract_coordinates_plddt(chain, max_len):
    """Returns a list of C-alpha coordinates for a chain"""
    pos = []
    plddt_scores = []
    for row, residue in enumerate(chain):
        pos_N_CA_C_O = []
        plddt_scores.append(residue["CA"].get_bfactor())
        for key in ['N', 'CA', 'C', 'O']:
            if key in residue:
                pos_N_CA_C_O.append(list(residue[key].coord))
            else:
                pos_N_CA_C_O.append([math.nan, math.nan, math.nan])

        pos.append(pos_N_CA_C_O)
        if len(pos) == max_len:
            break

    return pos, plddt_scores

def get_gpu_usage_smi():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                            stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split('\n')
    usage_info = []
    for i, line in enumerate(lines):
        gpu_util, mem_used, mem_total = line.split(', ')
        usage_info.append({
            'gpu_index': i,
            'gpu_utilization_percent': int(gpu_util),
            'memory_used_MB': int(mem_used),
            'memory_total_MB': int(mem_total),
        })
    return usage_info

def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_usage_gb = mem_info.rss / (1024 ** 3)  # 转换为 GB
    return memory_usage_gb

if __name__ == '__main__':
    print('for test')
    prot_seq, coords, scores = load_h5_file('./save_test/AF-G5EB01-F1-model_v4.h5')
    print(prot_seq, coords)
