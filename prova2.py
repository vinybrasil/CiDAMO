# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:39:23 2020

@author: vinic
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
#==========================================================================
#pro labelbinarizer

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
    
#==========================================================================

def EXPLORADOR(labels, data):
    
    print(data.info())
    data_array = np.array(data)
    #print(type(data_array))
    #print(data_array.shape)
    
    i = 0 
    while i<9:
        print(data_array[:,0])
        plt.scatter(data_array[:,i], labels)
        plt.show()
        i += 1
    
    
    #print(data.info())
    #print(data.describe(include='all'))
    #print(data['horario'].value_counts()) resolver com o labelBinrizer
    #print(data.head(20))
    
    
def DATA_BINARIZADOR_RESULTADOS(data):
    RESULTADO_BINARIO = []
    
    for item in data['retorno'].astype(str):
        if item == 'False':
            RESULTADO_BINARIO.append(0)
        elif item == 'True':
            RESULTADO_BINARIO.append(1)
        else:
            print("algo deu ruim")
            
    
    data['RESULTADO_BINARIO'] = RESULTADO_BINARIO
    return data
    

def separador(data):
    
    x = data.drop('retorno', axis=1)
    train_set, test_set = train_test_split(x, test_size=0.2, random_state=42)
    
    return train_set, test_set


def tira_resultados(data):

    DADOS_SEMLABELS = data.drop('RESULTADO_BINARIO', axis=1)
    DADOS_LABELS = data['RESULTADO_BINARIO']
    
    return DADOS_SEMLABELS, DADOS_LABELS
    

def arrumador_categoricas(data):
    
    CATEGORICAS = ['horario']
    
    NUMERICAS = data.drop('horario', axis=1)
    #print(NUMERICAS.head())
    num_pipeline = Pipeline([('selector', DataFrameSelector(list(NUMERICAS))),
                             ('std_scaler', StandardScaler())])
    
    cat_pipeline = Pipeline([('selector', DataFrameSelector(CATEGORICAS)),
                             ('label_binarizer', MyLabelBinarizer())])
    
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline), 
        ("cat_pipeline", cat_pipeline),
    ])
    DADOS_PREPARADOS = full_pipeline.fit_transform(data)
    
    return DADOS_PREPARADOS
                
def crialogisticRegression(data, labels):
    modelo = LogisticRegression().fit(data, labels)
    return modelo

def criarandomtree(data, labels):
    modelo = RandomForestClassifier().fit(data, labels)
    return modelo


def criaModeloApelao(data, labels):
    param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]}, 
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2, 3, 4]},
    ]   
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='r2')
    grid_search.fit(data, labels)
    return grid_search.best_estimator_


def validador(y_pred, labels, nome):
    print("\nResultados da ", nome, ": \n", 
           'acc = ', accuracy_score(labels, y_pred), "\n",
           'prec = ', precision_score(labels, y_pred), "\n",
           'recall = ', recall_score(labels, y_pred), "\n",
           'f1 = ', f1_score(labels, y_pred), "\n"
           'r2 = ', r2_score(labels, y_pred))

def main():
    
    data = pd.read_csv("data/ads_treino.csv")
    
    #antes de construir o modelo, a tarefa é arrumar os dados
    
    COM_RESULTADO_BINARIO = DATA_BINARIZADOR_RESULTADOS(data)
    
    DADOS_SEPARADOS_TREINO, DADOS_SEPARADOS_TESTE = separador(COM_RESULTADO_BINARIO)
    
    DADOS_TREINO_SEMLABEL, DADOS_TREINO_LABELS = tira_resultados(DADOS_SEPARADOS_TREINO)
    COM_TUDO_BINARIO_TREINO = arrumador_categoricas(DADOS_TREINO_SEMLABEL)
    
    DADOS_TESTE_SEMLABEL, DADOS_TESTE_LABELS = tira_resultados(DADOS_SEPARADOS_TESTE)
    COM_TUDO_BINARIO_TESTE = arrumador_categoricas(DADOS_TESTE_SEMLABEL)
    
    
    #EXPLORADOR(DADOS_TREINO_LABELS, DADOS_TREINO_SEMLABEL) 
    
    
    #construindo modelos:
    
    LOGISTICA = crialogisticRegression(COM_TUDO_BINARIO_TREINO, DADOS_TREINO_LABELS)
    y_pred_logistica = LOGISTICA.predict(COM_TUDO_BINARIO_TREINO)
    validador(y_pred_logistica, DADOS_TREINO_LABELS, "regressao logistica")
    #ja me da um 1 e 0s
    
  
    #o random forest classifier da 0 e 1, o regressor da em valores continuos
    RANDOM_ARVRE = criarandomtree(COM_TUDO_BINARIO_TREINO, DADOS_TREINO_LABELS)
    y_pred_arvre = RANDOM_ARVRE.predict(COM_TUDO_BINARIO_TREINO)
    validador(y_pred_arvre, DADOS_TREINO_LABELS, "random forest")
    
    MODELO_APELAO = criaModeloApelao(COM_TUDO_BINARIO_TREINO, DADOS_TREINO_LABELS)
    y_pred_apelao = MODELO_APELAO.predict(COM_TUDO_BINARIO_TREINO)
    validador(y_pred_apelao, DADOS_TREINO_LABELS, "random forest apelao")
    
if __name__ == "__main__":
    main()


