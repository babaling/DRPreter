import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import numpy as np
import pickle
from Model.DRPreter import DRPreter
from utils import *
from rdkit import DataStructs,Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearman
import argparse

dir = './Data/Similarity/'
dict_dir = './Data/Similarity/dict/'
with open(dict_dir + "cell_id2idx_dict", 'rb') as f:
    cell_id2idx_dict = pickle.load(f)
with open(dict_dir + "drug_name_cell_id_ic50", 'rb') as f:
    drug_name_cell_id_ic50 = pickle.load(f)
with open(dict_dir + "drug_idx_cell_idx_ic50", 'rb') as f:
    drug_idx_cell_idx_ic50 = pickle.load(f)
with open(dict_dir + "drug_name2smiles_dict", 'rb') as f:
    drug_name2smiles_dict = pickle.load(f)
with open(dict_dir + "drug_idx2smiles_dict", 'rb') as f:
    drug_idx2smiles_dict = pickle.load(f)
with open(dict_dir + "drug_name2idx_dict", 'rb') as f:
    drug_name2idx_dict = pickle.load(f)
with open(dict_dir + "cell_idx2id_dict", 'rb') as f:
    cell_idx2id_dict = pickle.load(f)
with open(dict_dir + "drug_idx2name_dict", 'rb') as f:
    drug_idx2name_dict = pickle.load(f)
cell_feature_normalized = np.load(rpath+f'Data/Cell/cell_feature_std_2369disjoint.npy', allow_pickle=True).item()


def computing_sim_matrix():
    """
    Construct similarity networks of cell and drug

    Returns:
        _dict_: cell similarity network, drug similarity network
    """    
    if os.path.exists(dict_dir + "cell_sim_matrix") and os.path.exists(dict_dir + "drug_sim_matrix"):
        with open(dict_dir+ "cell_sim_matrix", 'rb') as f:
            cell_sim_matrix = pickle.load(f)
        with open(dict_dir+ "drug_sim_matrix", 'rb') as f:
            drug_sim_matrix = pickle.load(f)
        return drug_sim_matrix, cell_sim_matrix
    
    cell_sim_matrix = np.zeros((len(cell_id2idx_dict), len(cell_id2idx_dict)))
    for i in range(len(cell_id2idx_dict)):
        for j in range(len(cell_id2idx_dict)):
            if i != j:
                cell_sim_matrix[i, j], _ = pearsonr(cell_feature_normalized[i], cell_feature_normalized[j])
                if cell_sim_matrix[i, j] < 0:
                    cell_sim_matrix[i, j] = 0
    
    drug_sim_matrix = np.zeros((len(drug_name2idx_dict), len(drug_name2idx_dict)))
    mi = [Chem.MolFromSmiles(drug_idx2smiles_dict[i]) for i in range(len(drug_name2idx_dict))]
    fps = [AllChem.GetMorganFingerprint(x, 4) for x in mi]
    for i in range(len(drug_name2idx_dict)):
        for j in range(len(drug_name2idx_dict)):
            if i != j:
                drug_sim_matrix[i, j] = DataStructs.DiceSimilarity(fps[i], fps[j])
    
    with open(dict_dir+ "cell_sim_matrix", 'wb') as f:
        pickle.dump(cell_sim_matrix, f)
    
    with open(dict_dir+ "drug_sim_matrix", 'wb') as f:
        pickle.dump(drug_sim_matrix, f)
    
    return drug_sim_matrix, cell_sim_matrix


def computing_knn(k):
    drug_sim_matrix, cell_sim_matrix = computing_sim_matrix()

    cell_sim_matrix_new = np.zeros_like(cell_sim_matrix)
    for u in range(len(cell_id2idx_dict)):
        v = cell_sim_matrix[u].argsort()[-6:-1]
        cell_sim_matrix_new[u][v] = cell_sim_matrix[u][v]
    
    drug_sim_matrix_new = np.zeros_like(drug_sim_matrix) 
    for u in range(len(drug_name2idx_dict)):
        v = drug_sim_matrix[u].argsort()[-6:-1]
        drug_sim_matrix_new[u][v] = drug_sim_matrix[u][v]
    
    cell_edges = np.argwhere(cell_sim_matrix_new >  0)
    drug_edges = np.argwhere(drug_sim_matrix_new >  0)
    
    with open(dir + "edge/drug_cell_edges_{}_knn".format(k), 'wb') as f:
        pickle.dump((drug_edges, cell_edges), f)


if __name__ == '__main__':
    computing_knn(5)