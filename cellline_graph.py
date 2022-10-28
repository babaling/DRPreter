import numpy as np
import pandas as pd
import os
import csv
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import sparse
import pickle
from tqdm import trange, tqdm


def ensp_to_hugo_map():
    with open(rpath+'Data/CELL/9606.protein.info.v11.5.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}

    return ensp_map


def save_cell_graph(gene_path, save_path, type):
    if os.path.exists(os.path.join(save_path, 'cell_feature_std_{}.npy'.format(type))):
        print('already exists!')
    else:
        # os.makedirs(save_path)
        exp = pd.read_csv(os.path.join(gene_path, 'CCLE_2369_EXP.csv'), index_col=0)
        index = exp.index
        columns = exp.columns

        scaler = StandardScaler()
        exp = scaler.fit_transform(exp)
        # cn = scaler.fit_transform(cn)

        imp_mean = SimpleImputer()
        exp = imp_mean.fit_transform(exp)

        exp = pd.DataFrame(exp, index=index, columns=columns)
        # cn = pd.DataFrame(cn, index=index, columns=columns)
        # mu = pd.DataFrame(mu, index=index, columns=columns)
        cell_names = exp.index

        cell_dict = {}

        for i in tqdm((cell_names)):

            # joint graph (without pathway)
            if type == 'joint':
                gene_list = exp.columns.to_list()
                gene_list = set()
                for pw in kegg:
                    for gene in kegg[pw]:
                        if gene in exp.columns.to_list():
                            gene_list.add(gene)
                gene_list = list(gene_list)
                cell_dict[i] = Data(x=torch.tensor([exp.loc[i, gene_list]], dtype=torch.float).T)
                # cell_dict[i] = Data(x=torch.tensor([cn.loc[i]], dtype=torch.float).T)
                # cell_dict[i] = Data(x=torch.tensor([mu.loc[i]], dtype=torch.float).T)
                # cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i]], dtype=torch.float).T)
                # cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i], me.loc[i]], dtype=torch.float).T)
                # cell_dict[i] = [np.array(exp.loc[i], dtype=np.float32), np.array(cn.loc[i], dtype=np.float32), np.array(mu.loc[i], dtype=np.float32)] # MLP용 코드
            
            # disjoint graph (with pathway)
            else:
                genes = exp.columns.to_list()
                x_mask = []
                x = []
                gene_list = {}
                for p, pw in enumerate(list(kegg)):
                    gene_list[pw] = []
                    for gene in kegg[pw]:
                        if gene in genes:
                            gene_list[pw].append(gene)
                            x_mask.append(p)
                    x.append(exp.loc[i, gene_list[pw]])
                x = pd.concat(x)
                cell_dict[i] = Data(x=torch.tensor([x], dtype=torch.float).T, x_mask=torch.tensor(x_mask, dtype=torch.int))

        print(cell_dict)
        np.save(os.path.join(save_path, 'cell_feature_std_{}.npy').format(type), cell_dict)
        print("finish saving cell data!")
        return gene_list


def get_STRING_edges(gene_path, ppi_threshold, type, gene_list):
    save_path = os.path.join(gene_path, 'edge_index_{}_{}.npy'.format(ppi_threshold, type))
    if not os.path.exists(save_path):
        # gene_list
        ppi = pd.read_csv(os.path.join(gene_path, 'CCLE_2369_{}.csv'.format(ppi_threshold)), index_col=0)

        # joint graph (without pathway)
        if type == 'joint':
            ppi = ppi.loc[gene_list, gene_list].values
            sparse_mx = sparse.csr_matrix(ppi).tocoo().astype(np.float32)
            edge_index = np.vstack((sparse_mx.row, sparse_mx.col))

        # disjoint graph (with pathway)
        else:
            edge_index = []
            for pw in gene_list:
                sub_ppi  = ppi.loc[gene_list[pw], gene_list[pw]]
                sub_sparse_mx = sparse.csr_matrix(sub_ppi).tocoo().astype(np.float32)
                sub_edge_index = np.vstack((sub_sparse_mx.row, sub_sparse_mx.col))
                edge_index.append(sub_edge_index)
            edge_index = np.concatenate(edge_index, 1)

        # Conserve edge_index
        print(len(edge_index[0]))
        np.save(
            os.path.join(rpath + 'Data/Cell/', 'edge_index_{}_{}.npy'.format(ppi_threshold, type)),
            edge_index)
    else:
        edge_index = np.load(save_path)

    return edge_index


if __name__ == '__main__':
    rpath = './'
    gene_path = rpath+'Data/Cell'
    save_path = rpath+'Data/Cell'
    with open(gene_path+'/34pathway_score990.pkl', 'rb') as file:
        kegg = pickle.load(file)

  
    type = 'disjoint'  # type = joint, disjoint, ...
    

    genelist = save_cell_graph(gene_path, save_path, type=type)
    get_STRING_edges(gene_path, ppi_threshold='PPI_990', type=type, gene_list=genelist)
