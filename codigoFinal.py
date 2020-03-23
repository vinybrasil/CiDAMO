# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:33:02 2020

@author: N13M4ND
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
def limpar_dados(data):
    data = data.dropna()
    return data

def divisor(data):
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)  
    return train_set, test_set

def arrumador_treinos(db):
    DADOS_SEMLABELS = db.drop('escore', axis=1)
    DADOS_LABELS = db['escore'].copy()
    return DADOS_SEMLABELS, DADOS_LABELS

def arrumador_categoricas(db):
    CATEGORICAS = ['grau_escolaridade']
    x = db.drop('estado_civil', axis=1)
    numericas = x.drop('grau_escolaridade',axis=1)
    print(numericas.info())
    num_pipeline = Pipeline([('selector', DataFrameSelector(list(numericas))),
                            ('std_scaler', StandardScaler())])
    
    cat_pipeline = Pipeline([('selector', DataFrameSelector(CATEGORICAS)),
                            ('label_binarizer', MyLabelBinarizer())])
    
    full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])
    
    DADOS_PREPARADOS = full_pipeline.fit_transform(x)
    return DADOS_PREPARADOS

def criaModelo(y, labels):
    return LinearRegression().fit(y, labels)
    
    
def main():
    DADOS_TREINO = pd.read_csv("dados/escore_treino.csv")
    DADOS_SEMNA = limpar_dados(DADOS_TREINO)
    RESPOSTA = input("É a fase de treino?\n")
    
    if RESPOSTA == "y":
        print("Fase de treinamento iniciada.")
        TREINO, TESTE = divisor(DADOS_SEMNA)   
        DADOS_SEMLABELS, DADOS_LABELS = arrumador_treinos(TREINO)
        DADOS_SEMLABELS_TESTE, DADOS_LABELS_TESTE = arrumador_treinos(TESTE)
        DADOS_PRONTOS = arrumador_categoricas(DADOS_SEMLABELS)
        print(DADOS_PRONTOS.shape) #tirei o estado civil
        model = criaModelo(DADOS_PRONTOS, DADOS_LABELS)
        yhat = model.predict(DADOS_PRONTOS)
        print(r2_score(DADOS_LABELS, yhat))
    else:
        print("Vejo que você está testando, então. ")
    
    

if __name__ == "__main__":
    main()


