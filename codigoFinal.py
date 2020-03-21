# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:33:02 2020

@author: N13M4ND
"""

from sklearn import linear_model
from sklearn import datasets
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

dados = pd.read_csv("dados/escore_treino.csv")
datas = dados.dropna()

z = datas['salario'].values
x = datas['experiência'].values
w = datas['idade'].values
e = datas['escore'].values
xlog = np.log(x + 1)
zlog = np.log(z + 1)

numericas = datas.select_dtypes('number')
categoricas = datas.select_dtypes('object')

categoricas = pd.get_dummies(categoricas[categoricas.columns])

feat = pd.concat([numericas, categoricas], axis=1)

feat.to_csv('features.csv')

print(list(feat))

e1 = feat['grau_escolaridade_fund. completo'].values
e2 = feat['grau_escolaridade_fund. incompleto'].values
e3 = feat['grau_escolaridade_médio completo'].values
e4 = feat['grau_escolaridade_médio incompleto'].values
e5 = feat['grau_escolaridade_pós-graduação'].values
e6 = feat['grau_escolaridade_superior completo'].values
e7 = feat['grau_escolaridade_superior incompleto'].values
s1 = feat['estado_civil_casado'].values
s2 = feat['estado_civil_divorciado'].values
s3 = feat['estado_civil_separado'].values
s4 = feat['estado_civil_solteiro'].values
s5 = feat['estado_civil_união estável formal'].values
s6 = feat['estado_civil_união estável informal'].values
s7 = feat['estado_civil_viúvo'].values

mod = smf.ols('e ~ zlog + w + xlog + e1 + e2 + e3 + e4 + e5 + e6 + e7 + s1 + s2 + s3 + s4 + s5 + s6 + s7', data=feat)
resultados = mod.fit()
print(resultados.summary())

teste = pd.read_csv("dados/escore_teste.csv")

numericas = teste.select_dtypes('number')
categoricas2 = teste.select_dtypes('object')
categoricas2 = pd.get_dummies(categoricas2[categoricas2.columns])
features2 = pd.concat([numericas, categoricas2], axis=1)
features2.to_csv('features2.csv')
features2['escore'] = ''
print(type(features2))
predito = resultados.predict(sm.add_constant(features2['escore'].values), transform=False)
print(predito)