import torch
import torch.nn as nn
from Model.DrugEncoder import DrugEncoder
from Model.CellEncoder import CellEncoder
from Model.Transformer import Transformer

class DRPreter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size

        self.num_feature = args.num_feature
        self.num_genes = args.num_genes
        self.cum_num_nodes = args.cum_num_nodes
        self.max_gene = args.max_gene
        self.n_pathways = args.n_pathways


        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dim_drug_cell = args.dim_drug_cell
        self.dropout_ratio = args.dropout_ratio
        self.trans = args.trans

        #  ---- (1) Drug branch ----
        self.DrugEncoder = DrugEncoder(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, self.dim_drug_cell),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio)
        )

        # ---- (2) Cell line branch ----
        self.CellEncoder = CellEncoder(self.num_feature, self.num_genes, self.layer_cell, self.dim_cell)


        if self.trans:
            self.cell_emb = nn.Sequential(
                nn.Linear(self.dim_cell * self.max_gene, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, self.dim_drug_cell),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio)
            )
            
            #self.token_emb = nn.Embedding(2, self.dim_drug_cell)
            self.Transformer = Transformer(d_model=self.dim_drug_cell, nhead=8, num_encoder_layers=1, dim_feedforward=self.dim_drug_cell)
            # reg_input = self.dim_drug_cell * (args.n_pathways+1)
            reg_input = self.dim_drug_cell*2
            reg_hidden = 512

        else:
            self.cell_emb = nn.Sequential(
                nn.Linear(self.dim_cell * self.num_genes, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, self.dim_drug_cell),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
            reg_input = self.dim_drug_cell*2
            reg_hidden = reg_input


        # ---- (3) Regression using cell embedding and drug emdbedding ----
        self.regression = nn.Sequential(
            nn.Linear(reg_input, reg_hidden),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(reg_hidden, reg_hidden),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(reg_hidden, 1)
        )


    def padding(self, x, mask):
        """
        [summary]
        Args:
            x (_type_): _description_
            mask (_type_): _description_

        Returns:
            x_pad: shape가 (batch size, total pathway num, max_gene * self.dim_cell)인 벡터
        """
 
        x = x.reshape(x.size(0), -1, self.dim_cell)                                        # x.shape: torch.Size([128, 4646, 8])
        x_pad = x.new_full((x.size(0), (mask[-1].item()+1) * self.max_gene, x.size(2)), 0) # x_pad.shape: torch.Size([128, 11934, 8])
        cum_num_nodes = self.cum_num_nodes.to(x.device)
        '''
        cum_num_nodes: tensor([   0,  142,  236,  373,  667,  741,  896, 1114, 1193, 1339, 1529, 1665,
        1958, 2125, 2198, 2430, 2489, 2728, 2837, 2941, 3026, 3152, 3253, 3474,
        3533, 3589, 3660, 3704, 4055, 4217, 4313, 4415, 4456, 4544, 4646],
        device='cuda:0')
        cum_num_nodes.shape: torch.Size([35])
        '''
        index = torch.arange(mask.size(0), dtype=torch.long, device=x.device)              # index.shape: torch.Size([4646]) ===> index: tensor([   0,    1,    2,  ..., 4643, 4644, 4645], device='cuda:0')
        index = (index - cum_num_nodes[mask]) + (mask * self.max_gene)                     # index.shape: torch.Size([4646]) ===> index: tensor([    0,     1,     2,  ..., 11682, 11683, 11684], device='cuda:0')
        x_pad[:, index, :] = x
        x_pad = x_pad.view(x.size(0), mask[-1].item() + 1, -1)                             # x_pad.shape: torch.Size([128, 34, 2808])

        return  x_pad                                                                      # shape (128, 34, max_gene * self.dim_cell)


    def aggregate(self, x_cell, x_drug, trans=False):

        if trans:
            '''
            x_drug shape: batch, 1, dim + (c_dim)
            x_cell shape: batch, path, dim + (c_dim)
            '''
            
            ''' method 1: pathway drug transformer -> pathway summation & drug : '''
            x_drug = x_drug.view(x_drug.size(0), 1, -1)        # x_drug.shape: torch.Size([128, 1, 256])

            ''' token embedding '''
            # x_drug_pos = self.token_emb(torch.zeros((x_drug.shape[0], x_drug.shape[1]), dtype=torch.long).to(device=x_drug.device))
            # x_cell_pos = self.token_emb(torch.ones((x_cell.shape[0], x_cell.shape[1]), dtype=torch.long).to(device=x_cell.device))
            # print(x_drug_pos.shape, x_cell_pos.shape)
            # x_cell += x_cell_pos
            # x_drug += x_drug_pos
            
            # -----
            x = torch.cat([x_cell, x_drug], 1)
            x, attn_score = self.Transformer(x)                # x.shape: torch.Size([128, 35, 256])
            x_cell = x[:, :-1, :].sum(dim=1)                   # x[:, :-1, :].shape: torch.Size([128, 34, 256]) ==> x_cell.shape: torch.Size([128, 256])
            x_drug_res = x[:, -1:, :]                          # x_drug.shape: torch.Size([128, 256])
            # print(x_drug.shape, x_drug_res.shape)
            x_drug += x_drug_res
            x_drug = x_drug.view(x_drug.size(0), -1)
            x = torch.cat([x_drug, x_cell], -1)
            # x = x.view(x.size(0), -1)
            # -----

            ''' method 2: only pathway summation & drug: 84.06%'''
            # x_cell = x_cell.sum(dim=1)                   # x[:, :-1, :].shape: torch.Size([128, 34, 256]) ==> x_cell.shape: torch.Size([128, 256])
            # x = torch.cat([x_drug, x_cell], -1)
            return x, attn_score
        else:
            ''' w/o transformer: squeezed all genes & drug: 83.11%'''
            x = torch.cat([x_drug, x_cell], -1)                # x.shape: torch.Size([128, 512])

            return x


    def forward(self, drug, cell):
        # ---- (1) forward drug ----
        x_drug = self.DrugEncoder(drug)
        x_drug = self.drug_emb(x_drug)

        # ---- (2) forward cell ----
        x_cell = self.CellEncoder(cell)
        mask = cell.x_mask[cell.batch==0].to(torch.long)

        x_cell = self.cell_emb(self.padding(x_cell, mask)) if self.trans else self.cell_emb(x_cell)

        # ---- (3) combine drug feature and cell line feature ----
        x, _ = self.aggregate(x_cell, x_drug, trans=self.trans)
        x = self.regression(x)

        return x
    
    
    def _embedding(self, drug, cell):
        """
        Get embeddings only
        """        
        # ---- (1) forward drug ----
        x_drug = self.DrugEncoder(drug)
        x_drug = self.drug_emb(x_drug)

        # ---- (2) forward cell ----
        x_cell = self.CellEncoder(cell)
        mask = cell.x_mask[cell.batch==0].to(torch.long)
        x_cell = self.cell_emb(self.padding(x_cell, mask)) if self.trans else self.cell_emb(x_cell)

        # ---- (3) Get cell line and drug embeddings ----
        return x_drug, x_cell


    def attn_score(self, drug, cell):
        # ---- (1) forward drug ----
        x_drug = self.DrugEncoder(drug)
        x_drug = self.drug_emb(x_drug)

        # ---- (2) forward cell ----
        x_cell = self.CellEncoder(cell)
        mask = cell.x_mask[cell.batch==0].to(torch.long)

        x_cell = self.cell_emb(self.padding(x_cell, mask)) if self.trans else self.cell_emb(x_cell)

        # ---- (3) combine drug feature and cell line feature ----
        _, attn_score = self.aggregate(x_cell, x_drug, trans=self.trans)

        return attn_score