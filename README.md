# DRPreter (Drug Response PREdictor and interpreTER)
DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided Graph Neural Networks and Transformer

![DRPreter](https://user-images.githubusercontent.com/68269057/198502117-785291dd-af73-40d3-8fed-0e8881404119.png)

## 1. Create cell-line data
```sh
python cell_line_graph.py
```


## 2. Create drug data
```sh
python drug_graph.py
```


## 3. Train a model
```sh
python main.py --mode train
```


## 4. Interpret model with trained model
```sh
python main.py --mode test --seed 0
```
