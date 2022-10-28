# DRPreter (Drug Response PREdictor and interpreTER)
DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided Graph Neural Networks and Transformer

[Overview.pdf](https://github.com/babaling/DRPreter/files/9885233/Overview.pdf)

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
python main.py --mode test --seed 0
```
