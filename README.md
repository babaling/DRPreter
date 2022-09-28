# DRPreter
DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided Graph Neural Networks and Transformer


###  source codes:
+ preprocess.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, performance measures and functions to draw loss, pearson by epoch.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.
+ training.py: train a GraphDRP model.
+ saliancy_map.py: run this to get saliency value.


## Dependencies
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
+ [Rdkit](https://www.rdkit.org/)
+ [Matplotlib](https://matplotlib.org/)
+ [Pandas](https://pandas.pydata.org/)
+ [Numpy](https://numpy.org/)
+ [Scipy](https://docs.scipy.org/doc/)


## 1. Create data in pytorch format
```sh
python preprocess.py
```


This returns file pytorch format (.pt) stored at data/processed including training, validation, test set.

## 2. Train a model
```sh
python main.py --mode train
```


## 3. Interpret model with trained model
```sh
python main.py --mode test
```
