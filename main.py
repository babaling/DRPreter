import os
import sys
import argparse
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import *
from Model.DRPreter import DRPreter
from Model.Similarity import Similarity
from torch_scatter import scatter_add


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--layer', type=int, default=3, help='Number of cell layers')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dim for cell')
    parser.add_argument('--layer_drug', type=int, default=3, help='Number of drug layers')
    parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug (default: 128)')
    parser.add_argument('--dim_drug_cell', type=int, default=256, help='hidden dim for drug and cell (default: 256)')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='Dropout ratio (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping (default: 10)')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--edge', type=str, default='STRING', help='STRING, BIOGRID') # BIOGRID: removed
    parser.add_argument('--string_edge', type=float, default=0.99, help='Threshold for edges of cell line graph')
    parser.add_argument('--dataset', type=str, default='2369disjoint', help='2369joint, 2369disjoint, COSMIC')
    parser.add_argument('--trans', type=bool, default=True, help='Use Transformer or not')
    parser.add_argument('--sim', type=bool, default=False, help='Construct homogeneous similarity networks or not')
    return parser.parse_args()


def main():
    args = arg_parse()
    args.device = 'cuda:{}'.format(args.device)
    rpath = './'
    result_path = rpath + 'Result/'
    
    print(f'seed: {args.seed}')
    set_random_seed(args.seed)
    
    edge_type = 'PPI_'+str(args.string_edge) if args.edge=='STRING' else args.edge
    edge_index = np.load(rpath+f'Data/Cell/edge_index_{edge_type}_{args.dataset}.npy')
    
    data = pd.read_csv(rpath+'Data/sorted_IC50_82833_580_170.csv')
    
    drug_dict = np.load(rpath+'Data/Drug/drug_feature_graph.npy', allow_pickle=True).item() # pyg format of drug graph
    cell_dict = np.load(rpath+f'Data/Cell/cell_feature_std_{args.dataset}.npy', allow_pickle=True).item() # pyg data format of cell graph

    example = cell_dict['ACH-000001']
    args.num_feature = example.x.shape[1] # 1
    args.num_genes = example.x.shape[0] # 4646
    # print(f'num_feature: {args.num_feature}, num_genes: {args.num_genes}')
    # sys.exit('Bye!')
            
    if 'disjoint' in args.dataset:
        gene_list = scatter_add(torch.ones_like(example.x.squeeze()), example.x_mask.to(torch.int64)).to(torch.int)
        args.max_gene = gene_list.max().item()
        args.cum_num_nodes = torch.cat([gene_list.new_zeros(1), gene_list.cumsum(dim=0)], dim=0)
        args.n_pathways = gene_list.size(0)
        print('num_genes:{}, num_edges:{}'.format(args.num_genes, len(edge_index[0])))
        print('gene distribution: {}'.format(gene_list))
        print('mean degree:{}'.format(len(edge_index[0]) / args.num_genes))
    else:
        print('num_genes:{}, num_edges:{}'.format(args.num_genes, len(edge_index[0])))
        print('mean degree:{}'.format(len(edge_index[0]) / args.num_genes))
        
        
    # ---- [1] Pathway + Transformer ----
    if args.sim == False:
        train_loader, val_loader, test_loader = load_data(data, drug_dict, cell_dict, torch.tensor(edge_index, dtype=torch.long), args)
        print('total: {}, train: {}, val: {}, test: {}'.format(len(data), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
        
        model = DRPreter(args).to(args.device)
        # print(model)
        
        
    # ---- [2] Add similarity information after obtaining embeddings ----
    else:
        train_loader, val_loader, test_loader = load_sim_data(data, args)
        print('total: {}, train: {}, val: {}, test: {}'.format(len(data), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
        drug_nodes_data, cell_nodes_data, drug_edges, cell_edges = load_sim_graph(torch.tensor(edge_index, dtype=torch.long), args)

        model = Similarity(drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, args).to(args.device)
        # print(model)

        
# -----------------------------------------------------------------
            
            
    if args.mode == 'train':
        result_col = ('mse\trmse\tmae\tpcc\tscc')
        
        result_type = 'results_sim' if args.sim==True else 'results'
        results_path = get_path(args, result_path, result_type=result_type)
        
        with open(results_path, 'w') as f:
            f.write(result_col + '\n')
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        state_dict_name = f'{rpath}weights/weight_sim_seed{args.seed}.pth' if args.sim==True else f'{rpath}weights/weight_seed{args.seed}.pth'
        stopper = EarlyStopping(mode='lower', patience=args.patience, filename=state_dict_name)

        for epoch in range(1, args.epochs + 1):
            print(f"===== Epoch {epoch} =====")
            train_loss = train(model, train_loader, criterion, opt, args)

            mse, rmse, mae, pcc, scc, _ = validate(model, val_loader, args)
            results = [epoch, mse, rmse, mae, pcc, scc]
            save_results(results, results_path)
            
            print(f"Validation mse: {mse}")
            test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(model, test_loader, args)
            print(f"Test mse: {test_MSE}")
            early_stop = stopper.step(mse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Best epoch: {}'.format(epoch-stopper.counter))

        stopper.load_checkpoint(model)

        train_MSE, train_RMSE, train_MAE, train_PCC, train_SCC, _ = validate(model, train_loader, args)
        val_MSE, val_RMSE, val_MAE, val_PCC, val_SCC, _ = validate(model, val_loader, args)
        test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(model, test_loader, args)

        print('-------- DRPreter -------')
        print(f'sim: {args.sim}')
        print(f'##### Seed: {args.seed} #####')
        print('\t\tMSE\tRMSE\tMAE\tPCC\tSCC')
        print('Train result: {}\t{}\t{}\t{}\t{}'.format(r4(train_MSE), r4(train_RMSE), r4(train_MAE), r4(train_PCC), r4(train_SCC)))
        print('Val result: {}\t{}\t{}\t{}\t{}'.format(r4(val_MSE), r4(val_RMSE), r4(val_MAE), r4(val_PCC), r4(val_SCC)))
        print('Test result: {}\t{}\t{}\t{}\t{}'.format(r4(test_MSE), r4(test_RMSE), r4(test_MAE), r4(test_PCC), r4(test_SCC)))
        df.to_csv(get_path(args, result_path, result_type=result_type+'_df', extension='csv'), sep='\t', index=0)

        
    elif args.mode == 'test':
        state_dict_name = f'{rpath}weights/weight_sim_seed{args.seed}.pth' if args.sim==True else f'{rpath}weights/weight_seed{args.seed}.pth'
        model.load_state_dict(torch.load(rpath + state_dict_name, map_location=args.device)['model_state_dict'])


#         '''Get embeddings of specific drug and cell line pair'''
#         drug_name, cell_name = 'Bortezomib', 'ACH-000137' # 8MGBA
#         drug_emb, cell_emb = embedding(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
#         print(drug_emb, cell_emb)
        
        
        ''' Test results only '''
        test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(model, test_loader, args)
        print('-------- DRPreter -------')
        print(f'sim: {args.sim}')
        print(f'##### Seed: {args.seed} #####')
        print('\t\tMSE\tRMSE\tMAE\tPCC\tSCC')
        print('Test result: {}\t{}\t{}\t{}\t{}'.format(r4(test_MSE), r4(test_RMSE), r4(test_MAE), r4(test_PCC), r4(test_SCC)))
        
        
        '''GradCAM'''
        # ----- (1) Calculate gradient-based importance score for one cell line-drug pair -----        
        # drug_name, cell_name = 'Dihydrorotenone', 'ACH-001374'
        # gradcam_path =  get_path(args, rpath + 'GradCAM/', result_type=f'{drug_name}_{cell_name}_gradcam', extension='csv')
        
        # gene_dict = np.load(rpath + 'Data/Cell/cell_idx2gene_dict.npy', allow_pickle=True)
        
        # # Save importance score
        # sorted_cell_node_importance, indices = gradcam(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        # idx2gene = [gene_dict[idx] for idx in indices]
        
        # sorted_cell_node_importance = list(sorted_cell_node_importance.cpu().detach().numpy())
        # indice = list(indices)
        
        # df = pd.DataFrame((zip(sorted_cell_node_importance, indice, idx2gene)), columns=['cell_node_importance','indice','idx2gene'])
        # # df.to_csv(gradcam_path, index=False)
        # print(*list(df['idx2gene'])[:30])
        
        # ----- (2) Calculate scores from total test set in 'inference.csv' -----
        # data = pd.read_excel(f'inference_seed{args.seed}.xlsx', sheet_name='test')
        # name = data[['Drug name', 'DepMap_ID']]
        
        # gene_dict = np.load(rpath + 'Data/Cell/cell_idx2gene_dict.npy', allow_pickle=True)
        
        # total_gene_df = pd.Series(list(range(len(data))))
        # for i in tqdm(range(len(data))):
        #     drug_name, cell_name = name.iloc[i]
        #     _, indices = gradcam(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        #     idx2gene = [gene_dict[idx] for idx in indices]
        #     gene_df = pd.DataFrame(idx2gene)
        #     total_gene_df.loc[i] = ', '.join(list(gene_df.drop_duplicates(keep='first')[0])[:5])
        
        # data['Top5 genes'] = total_gene_df
        # data.to_excel(f'inference_seed{args.seed}_gradcam.xlsx', sheet_name='test')
        
            
        '''Visualize pathway-drug self-attention score from Transformer'''
        # ----- (1) For one cell line - drug pair -----
        
        # drug_name, cell_name = 'Rapamycin', 'ACH-000019'

        # # print(cell_name)
        # attn_score = attention_score(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        # print(f'attn_score: {attn_score}')
        # print(f'attn_score.shape: {attn_score.shape}') # attn_score.shape: torch.Size([1, 35, 35])
        # # print(torch.sum(attn_score, axis=1))
        # with open(rpath+'Data/Cell/34pathway_score990.pkl', 'rb') as file:
        #     pathway_names = pickle.load(file).keys()
        # tks = [p[5:] for p in list(pathway_names)]
        # tks.append(drug_name)
        # # print(tks)
        # draw_pair_heatmap(attn_score, drug_name, cell_name, tks, args)
        
        # ----- (2) Heatmap of all cell lines of one drug -----
        
        # drug_name = 'Rapamycin'
        # data = pd.read_csv(f'./Data/{drug_name}.csv')
        # cell_list = list(data['DepMap_ID'])
        
        # result_dict = {}
        # total_result = np.full(35, 0)
        # for cell_name in tqdm(cell_list):
        #     attn_score = attention_score(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
        #     print(attn_score.shape)
        #     attn_score = torch.squeeze(attn_score).cpu().detach().numpy()
        #     print(np.sum(attn_score, axis=1))
        #     result_dict[cell_name] = attn_score[-1, :] # (35, 1)
        #     total_result = np.vstack([total_result, attn_score[-1, :]])
        
        # with open(rpath+'Data/Cell/34pathway_score990.pkl', 'rb') as file:
        #     pathway_names = pickle.load(file).keys()
        # xtks = [p[5:] for p in list(pathway_names)]
        # xtks.append(drug_name)
        # total_result = total_result[1:,:-1]
        # draw_drug_heatmap(total_result, drug_name, xtks, cell_list, args)
        

        '''Interpolation of unknown values'''
#         inference(model, drug_dict, cell_dict, edge_index, f'inference_seed{args.seed}.xlsx', args)
        
        
if __name__ == "__main__":
    main()