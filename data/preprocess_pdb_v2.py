"""
In this v2, we have the following updates:
1 extend the atom types to 'CA','N','C','O','CB'#,'CG1','CG2' 
2 add DSSP 8-state secondary structure
3 add SASA values
4 node_degree
5 edge_index for cutoff = 12
6 edge_dis each edge has 25D d_ij (only saved), 1/d_ij**2,1/d_ij**4,1/d_ij**6 for all CA-CA, CA-C, CA-N, CA-O,CA-CB       O-N,O-C.O-O,O-CA,O-CB,                 N-O, N-C,N-N, N-CA, N-CB,              C-C, C-N, C-O, C-CA, C-CB,                    CB-C, CB-N, CB-O, CB-CA,CB-CB 
7 edge_localorientation each edge has 75D 
refer to ppt
"""
import argparse
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
#from Bio import pairwise2
import math
import h5py
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import Manager
from Bio.PDB.DSSP import DSSP
from Bio.PDB.SASA import ShrakeRupley
import Bio
import pandas as pd
import numpy as np


standard_aa_names = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    ]

def write_h5_file(file_path, pad_seq, n_ca_c_o_coord, plddt_scores,ss,sasa,node_degree,edge_index_0,edge_index_1,edge_dis,edge_localorientation):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('seq', data=pad_seq)
        f.create_dataset('N_CA_C_O_coord', data=n_ca_c_o_coord)
        f.create_dataset('plddt_scores', data=plddt_scores)
        f.create_dataset('ss', data=ss)
        f.create_dataset('sasa', data=sasa)
        f.create_dataset('node_degree', data=node_degree)
        f.create_dataset('edge_index', data=[edge_index_1,edge_index_0]) #to be consistent with edge_index_tensor = radius_graph(torch.tensor(X_CA), r=12)
        f.create_dataset('edge_dis', data=edge_dis) #can select to use first 3D, edge_dis[edge_index][0]
        f.create_dataset('edge_localorientation', data=edge_localorientation) #can select to use first 3D  edge_localorientation[edge_index][:3]

def preprocess_file(file_path, max_len, save_path, report_dict):
    '''for debug if stopped,skip existing files, only for alphafold4 predicted results at the same outputfile
    file_path_base = os.path.basename(file_path)
    outputfile = os.path.join(save_path, os.path.splitext(file_path_base)[0] + f"_chain_id_A.h5")
    if os.path.exists(outputfile):
        return 1
    '''
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    model = structure[0] #only use the fist model
    #delete irregular residue atoms
    dssp = DSSP(model, file_path, dssp='dssp' )
    for chain in model:
        delete_REScluster=[]
        for residue_index, residue in enumerate(chain):
            #print(residue.id)
            if residue.get_resname() not in standard_aa_names:
                delete_REScluster.append(residue.id)
            
            #regular residues should have all of these, if not just detach this residue
            if 'CA' not in residue or "C" not in residue or "N" not in residue or "O" not in residue:
                delete_REScluster.append(residue.id)
            
            if residue_index>=max_len: #longer residues are all removed from sequence
                delete_REScluster.append(residue.id)
        
        if delete_REScluster!=[]:
            delete_REScluster = set(delete_REScluster)
            for delete_res in delete_REScluster:
                chain.detach_child(delete_res)
    
    #neighbor search
    atoms  = Bio.PDB.Selection.unfold_entities(model, 'A')
    ns = Bio.PDB.NeighborSearch(atoms)
    sr = ShrakeRupley()
    sr.compute(model, level="R")
    for chain in model:
        chain_id = chain.id
        poly=Bio.PDB.Polypeptide.Polypeptide(chain)
        sequence_list =poly.get_sequence()
        protein_seq = "".join(sequence_list)
        pos = []
        plddt_scores = []
        ss = []
        sasa = []
        node_degree = []
        has_ambiguous_residue = False
        residue_list=[residue.id for residue in chain]
        matrix = [[i,chain.id,j,l] for i,(j,(k,l,m)) in enumerate(zip(sequence_list ,residue_list))]
        #matrix[0]: [0, 'A', 'M', 1] i residue_index  l next index, index+1 j residue_type
        matrix = pd.DataFrame(matrix)
        edge_localorientation = []
        edge_dis = []
        edge_index_0 = []
        edge_index_1 = []
        for residue_index, residue in enumerate(chain):
            pos_N_CA_C_O = []
            try:
                plddt_scores.append(residue["CA"].get_bfactor())
            except KeyError:
                plddt_scores.append(0)
            for key in ['N', 'CA', 'C', 'O']:
                if key in residue:
                    pos_N_CA_C_O.append(list(residue[key].coord))
                else:
                    pos_N_CA_C_O.append([math.nan, math.nan, math.nan])
            pos.append(pos_N_CA_C_O)
            ss.append(dssp[chain.id,residue.id][2]) #add 8 state SS 
            sasa.append(residue.sasa)
            #extract neighbor atoms around CA
            NA = ns.search(chain[int(matrix.iloc[residue_index,3])]['CA'].coord, 12)  ##Neighbor Atoms
            
            All_CA = [atom for atom in NA if atom.id=='CA']   # atom.get_parent().id[1]
            #All_CA: A list of all neighboring alpha carbon atoms within the specified radius.
            #add node degree feature
            if len(All_CA) == 1:
                node_degree.append([0,10000])
            else:
                node_degree.append([len(All_CA)-1,1/(len(All_CA)-1)])
            
            edge_index_0,edge_index_1,edge_dis,edge_localorientation = cal_edge_feature_perCA(residue_index,residue,All_CA,matrix,chain,edge_index_0,edge_index_1,edge_dis,edge_localorientation)
            
            if len(pos) == max_len:
                break
        
        if has_ambiguous_residue:
            continue
        
        pad_seq = protein_seq[:max_len]
        
        file_path_base = os.path.basename(file_path)
        if ".pdb" in os.path.splitext(file_path_base)[1]:
                outputfile = os.path.join(save_path, os.path.splitext(file_path_base)[0] + f"_chain_id_{chain_id}.h5")
        else:
                outputfile = os.path.join(save_path, file_path_base + f"_chain_id_{chain_id}.h5")
        
        report_dict['h5_processed'] += 1
        write_h5_file(outputfile, pad_seq, pos, plddt_scores,ss,sasa,node_degree,edge_index_0,edge_index_1,edge_dis,edge_localorientation)


def cal_edge_feature_perCA(i,residue,All_CA,matrix,chain,edge_index_0,edge_index_1,edge_dis,edge_localorientation):
    #i residue index
    #orthognal for this CA
    #All_CA
    CA_coord = residue['CA'].coord #should equal chain[int(matrix .iloc[i,3])]['CA'].coord
    C_coord = residue['C'].coord
    N_coord = residue['N'].coord
    O_coord = residue['O'].coord
    CA_C = C_coord - CA_coord #x
    CA_N = N_coord - CA_coord #y
    #CA_O = O_coord - CA_coord #Z #no need
    orthognal = np.cross(CA_C, CA_N) #Z
    for CA in All_CA: 
        #Ca is not the center ca and they belong to the same chain
        if (CA.coord != chain[int(matrix .iloc[i,3])]['CA'].coord).any() and CA.get_parent().get_parent().id == chain.id :
            each_edge_dis = []
            each_edge_localorientation = []
            #CA -> CA
            CA_CA_dis,CA_CA_orientation= cal_dis_orientation(CA.coord,CA_coord,CA_C,CA_N,orthognal)
            """
            ## CA -> C
            CA_C_dis,CA_C_orientation= cal_dis_orientation(CA.get_parent()['C'].coord,CA_coord,CA_C,CA_N,orthognal)
            ## CA -> N
            CA_N_dis,CA_N_orientation= cal_dis_orientation(CA.get_parent()['N'].coord,CA_coord,CA_C,CA_N,orthognal)
            ## CA->O 
            CA_O_dis,CA_O_orientation= cal_dis_orientation(CA.get_parent()['O'].coord,CA_coord,CA_C,CA_N,orthognal)
            
            ##CA_CB
            if CA.get_parent().resname !='GLY':
                CA_CB_dis,CA_CB_orientation= cal_dis_orientation(CA.get_parent()['CB'].coord,CA_coord,CA_C,CA_N,orthognal)
            else:
                CA_CB_dis = CA_CA_dis
                CA_CB_orientation = CA_CA_orientation
            
            ## O -> N, where O is from the central residue and N is from the neighbor residue
            O_N_dis,O_N_orientation= cal_dis_orientation(CA.get_parent()['N'].coord,O_coord,CA_C,CA_N,orthognal)
            ## O (center) ->C 
            O_C_dis,O_C_orientation= cal_dis_orientation(CA.get_parent()['C'].coord,O_coord,CA_C,CA_N,orthognal)
            ## O ->O
            O_O_dis,O_O_orientation= cal_dis_orientation(CA.get_parent()['O'].coord,O_coord,CA_C,CA_N,orthognal)
            ## O (center)->CA
            O_CA_dis,O_CA_orientation= cal_dis_orientation(CA.coord,O_coord,CA_C,CA_N,orthognal)
            ## o (center)->CB
            if CA.get_parent().resname !='GLY':
                O_CB_dis,O_CB_orientation= cal_dis_orientation(CA.get_parent()['CB'].coord,O_coord,CA_C,CA_N,orthognal)
                #O_CB_dis = np.linalg.norm( CA.get_parent()['CB'].coord - O_coord )
                #O_CB_orientation = [np.dot( CA.get_parent()['CB'].coord - O_coord , CA_C )/np.linalg.norm( CA_C ), np.dot( CA.get_parent()['CB'].coord - O_coord , CA_N )/np.linalg.norm( CA_N ), np.dot( CA.get_parent()['CB'].coord - O_coord , orthognal )/np.linalg.norm( orthognal )]
            else:
                O_CB_dis = O_CA_dis
                O_CB_orientation = O_CA_orientation
            
            ## N(center) -> O, where N is from the central residue and O is from the neighbor residue
            N_O_dis,N_O_orientation= cal_dis_orientation(CA.get_parent()['O'].coord,N_coord,CA_C,CA_N,orthognal)
            ## N -> C
            N_C_dis,N_C_orientation= cal_dis_orientation(CA.get_parent()['C'].coord,N_coord,CA_C,CA_N,orthognal)
            ## N -> N
            N_N_dis,N_N_orientation= cal_dis_orientation(CA.get_parent()['N'].coord,N_coord,CA_C,CA_N,orthognal)
            ## N -> CA
            N_CA_dis,N_CA_orientation= cal_dis_orientation(CA.coord,N_coord,CA_C,CA_N,orthognal)
            ## N ->CB
            if CA.get_parent().resname !='GLY':
                N_CB_dis,N_CB_orientation= cal_dis_orientation(CA.get_parent()['CB'].coord,N_coord,CA_C,CA_N,orthognal)
            else:
                N_CB_dis = N_CA_dis
                N_CB_orientation = N_CA_orientation
            
            ## C -> C, 
            C_C_dis,C_C_orientation= cal_dis_orientation(CA.get_parent()['C'].coord,C_coord,CA_C,CA_N,orthognal)
            ## C -> N
            C_N_dis,C_N_orientation= cal_dis_orientation(CA.get_parent()['N'].coord,C_coord,CA_C,CA_N,orthognal)
            ## C -> O
            C_O_dis,C_O_orientation= cal_dis_orientation(CA.get_parent()['O'].coord,C_coord,CA_C,CA_N,orthognal)
            ## C -> CA
            C_CA_dis,C_CA_orientation= cal_dis_orientation(CA.coord,C_coord,CA_C,CA_N,orthognal)
            ## C ->CB
            if CA.get_parent().resname !='GLY':
                C_CB_dis,C_CB_orientation= cal_dis_orientation(CA.get_parent()['CB'].coord,C_coord,CA_C,CA_N,orthognal)
            else:
                C_CB_dis = C_CA_dis
                C_CB_orientation = C_CA_orientation
            
            ## CB(center) ->C,N,O,CA,
            if chain[int(matrix .iloc[i,3])].resname !='GLY':
                assert not (chain[int(matrix.iloc[i, 3])]['CB'].coord != residue['CB'].coord).any(), "CB coordinates should identical"
                #CB_C_dis,CB_C_orientation= cal_dis_orientation(CA.get_parent()['C'].coord,chain[int(matrix.iloc[i,3])]['CB'].coord,CA_C,CA_N,orthognal)
                CB_C_dis,CB_C_orientation= cal_dis_orientation(CA.get_parent()['C'].coord,residue['CB'].coord,CA_C,CA_N,orthognal)
                CB_N_dis,CB_N_orientation= cal_dis_orientation(CA.get_parent()['N'].coord,residue['CB'].coord,CA_C,CA_N,orthognal)
                CB_O_dis,CB_O_orientation= cal_dis_orientation(CA.get_parent()['O'].coord,residue['CB'].coord,CA_C,CA_N,orthognal)
                CB_CA_dis,CB_CA_orientation= cal_dis_orientation(CA.get_parent()['CA'].coord,residue['CB'].coord,CA_C,CA_N,orthognal)
                if CA.get_parent().resname !='GLY':
                    CB_CB_dis,CB_CB_orientation= cal_dis_orientation(CA.get_parent()['CB'].coord,residue['CB'].coord,CA_C,CA_N,orthognal)
                else:
                    CB_CB_dis = CB_CA_dis
                    CB_CB_orientation = CB_CA_orientation
            else:
                CB_C_dis = CA_C_dis #np.linalg.norm( CA.get_parent()['C'].coord - CA_coord )
                CB_C_orientation = CA_C_orientation #[np.dot( (CA.get_parent()['C'].coord - CA_coord), CA_C )/np.linalg.norm( CA_C ), np.dot( (CA.get_parent()['C'].coord - CA_coord), CA_N )/np.linalg.norm( CA_N ), np.dot( (CA.get_parent()['C'].coord - CA_coord), orthognal )/np.linalg.norm( orthognal )]
                CB_N_dis = CA_N_dis #np.linalg.norm( CA.get_parent()['N'].coord - CA_coord )
                CB_N_orientation = CA_N_orientation #[np.dot( (CA.get_parent()['N'].coord - CA_coord), CA_C )/np.linalg.norm( CA_C ), np.dot( (CA.get_parent()['N'].coord - CA_coord), CA_N )/np.linalg.norm( CA_N ), np.dot( (CA.get_parent()['N'].coord - CA_coord), orthognal )/np.linalg.norm( orthognal )]    
                CB_O_dis = CA_O_dis #np.linalg.norm( CA.get_parent()['O'].coord - CA_coord )
                CB_O_orientation = CA_O_orientation #[np.dot( (CA.get_parent()['O'].coord - CA_coord), CA_C )/np.linalg.norm( CA_C ), np.dot( (CA.get_parent()['O'].coord - CA_coord), CA_N )/np.linalg.norm( CA_N ), np.dot( (CA.get_parent()['O'].coord - CA_coord), orthognal )/np.linalg.norm( orthognal )]    
                CB_CA_dis = CA_CA_dis
                CB_CA_orientation = CA_CA_orientation
                if CA.get_parent().resname !='GLY':
                    CB_CB_dis,CB_CB_orientation= cal_dis_orientation(CA.get_parent()['CB'].coord,CA_coord,CA_C,CA_N,orthognal)
                else:
                    CB_CB_dis = CA_CA_dis
                    CB_CB_orientation = CA_CA_orientation
            """
            #each_edge_attributes.extend([CA_CA_dis,1/CA_CA_dis**2,1/CA_CA_dis**4,1/CA_CA_dis**6])
            each_edge_dis.extend([CA_CA_dis])
            each_edge_localorientation.extend(CA_CA_orientation)
            """
            each_edge_dis.extend([CA_C_dis])
            each_edge_localorientation.extend(CA_C_orientation)
            each_edge_dis.extend([CA_O_dis])
            each_edge_localorientation.extend(CA_O_orientation)
            each_edge_dis.extend([CA_N_dis])
            each_edge_localorientation.extend(CA_N_orientation)
            each_edge_dis.extend([CA_CB_dis])
            each_edge_localorientation.extend(CA_CB_orientation)
            each_edge_dis.extend([O_N_dis])
            each_edge_localorientation.extend(O_N_orientation)
            each_edge_dis.extend([O_C_dis])
            each_edge_localorientation.extend(O_C_orientation)
            each_edge_dis.extend([O_CA_dis])
            each_edge_localorientation.extend(O_CA_orientation)
            each_edge_dis.extend([O_O_dis])
            each_edge_localorientation.extend(O_O_orientation)
            each_edge_dis.extend([O_CB_dis])
            each_edge_localorientation.extend(O_CB_orientation)
            each_edge_dis.extend([N_O_dis])
            each_edge_localorientation.extend(N_O_orientation)
            each_edge_dis.extend([N_C_dis])
            each_edge_localorientation.extend(N_C_orientation)
            each_edge_dis.extend([N_CA_dis])
            each_edge_localorientation.extend(N_CA_orientation)
            each_edge_dis.extend([N_N_dis])
            each_edge_localorientation.extend(N_N_orientation)
            each_edge_dis.extend([N_CB_dis])
            each_edge_localorientation.extend(N_CB_orientation)
            each_edge_dis.extend([C_O_dis])
            each_edge_localorientation.extend(C_O_orientation)
            each_edge_dis.extend([C_N_dis])
            each_edge_localorientation.extend(C_N_orientation)
            each_edge_dis.extend([C_C_dis])
            each_edge_localorientation.extend(C_C_orientation)
            each_edge_dis.extend([C_CA_dis])
            each_edge_localorientation.extend(C_CA_orientation)
            each_edge_dis.extend([C_CB_dis])
            each_edge_localorientation.extend(C_CB_orientation)
            each_edge_dis.extend([CB_O_dis])
            each_edge_localorientation.extend(CB_O_orientation)
            each_edge_dis.extend([CB_N_dis])
            each_edge_localorientation.extend(CB_N_orientation)
            each_edge_dis.extend([CB_C_dis])
            each_edge_localorientation.extend(CB_C_orientation)
            each_edge_dis.extend([CB_CA_dis])
            each_edge_localorientation.extend(CB_CA_orientation)
            each_edge_dis.extend([CB_CB_dis])
            each_edge_localorientation.extend(CB_CB_orientation)
            """
            ##beginnode to endnode 
            endnode = matrix[ matrix.iloc[:,3] == CA.get_parent().id[1] ].iloc[0,0]
            #each_edge_attributes.extend([sin(matrix.iloc[endnode,3] - matrix.iloc[i,3] ),cos(matrix.iloc[endnode,3] - matrix.iloc[i,3])])
            edge_index_0.append(i)
            edge_index_1.append(endnode)
            edge_dis.append(each_edge_dis)
            edge_localorientation.append(each_edge_localorientation)
        
    
    return edge_index_0,edge_index_1,edge_dis,edge_localorientation

def cal_dis_orientation(node_neighbor,node_center,CA_C,CA_N,orthognal):
    dis = np.linalg.norm(node_neighbor-node_center) #scalar value
    orientation = [
               np.dot( (node_neighbor - node_center), CA_C )/np.linalg.norm( CA_C ), 
               np.dot( (node_neighbor - node_center), CA_N )/np.linalg.norm( CA_N ), 
               np.dot( (node_neighbor - node_center), orthognal )/np.linalg.norm( orthognal )] #dx,dy,dz
    
    return dis,orientation
    


def main():
    parser = argparse.ArgumentParser(description='Processing PDB files.')
    parser.add_argument('--data', default='./test_data', help='Path to PDB files.')
    parser.add_argument('--max_len', default=1024, type=int, help='Max sequence length to consider.')
    parser.add_argument('--save_path', default='./save_test/', help='Path to output.')
    parser.add_argument('--max_workers', default=16,type=int,
                        help='Set the number of workers for parallel processing.')
    args = parser.parse_args()

    data_path = [os.path.join(args.data, path) for path in os.listdir(args.data) if os.path.isfile(os.path.join(args.data, path))]# and path.endswith('pdb')]
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with Manager() as manager:
        report_dict = manager.dict({'protein_complex': 0, 'no_chain_id_a': 0, 'h5_processed': 0,
                                    'single_amino_acid': 0, 'error': 0})
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(preprocess_file, file_path, args.max_len, args.save_path, report_dict): file_path for file_path in data_path}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred while processing {file_path}: {exc} {type(exc)}")
                    report_dict['error'] += 1
        print(dict(report_dict))


if __name__ == '__main__':
    main()



#python preprocess_pdb_v2.py --data /data/duolin/alphafold4 --max_len 512 --save_path /data/duolin/swiss-pos-512-v2 --max_workers 8
#
#python preprocess_pdb_v2.py --data /data/duolin/train_PLK1/datasets/structuresAs --max_len 512 --save_path /data/duolin/train_PLK1/datasets/structuresAs_hdf5_v2 --max_workers 8
