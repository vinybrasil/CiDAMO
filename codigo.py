# -*- coding: utf-8 -*-
"""
@author: N13M4ND
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
dados = pd.read_csv("dados/escore_treino.csv")

#print(dados.head(10))

print(list(dados))
#print(dados.describe())

#arrumar primeiro as variaveis categoricas: estado civil e grau de escolaridade

data = dados.dropna()
print(data.describe())

#plt.plot se quiser um grafico de linhas
#plt.scatter(data['salario'], data['escore'])
#plt.show()
'''
ohe = OneHotEncoder()
escola = ohe.fit_transform(data[['grau_escolaridade']])
print(ohe.categories_)
status = ohe.fit_transform(data[['estado_civil']])
print(ohe.categories_)
'''


z = data['salario'].values
x = data[['experiência']].values
w = data[['idade']].values
e = data[['escore']]
xlog = np.log(x + 1)
zlog = np.log(z + 1)

resultados = smf.ols('e ~ z + w + x', data=data).fit()
print(resultados.summary())

#os coef taos bons, mas o R^2 ta baixo(0.658), logo falta outras variaveis
#tentar o onehotencoding:

numericas = data.select_dtypes('number')
categoricas = data.select_dtypes('object')

categoricas = pd.get_dummies(categoricas[categoricas.columns])

features = pd.concat([numericas, categoricas], axis=1)
#print(features.head())

#salvar para csv:
features.to_csv('features.csv')

#agora ta assim as columns: ,Id,salario,idade,experiência,escore,
#grau_escolaridade_fund. completo,grau_escolaridade_fund. incompleto,
#grau_escolaridade_médio completo,grau_escolaridade_médio incompleto,
#grau_escolaridade_pós-graduação,grau_escolaridade_superior completo,
#grau_escolaridade_superior incompleto,estado_civil_casado,
#estado_civil_divorciado,estado_civil_separado,estado_civil_solteiro,
#estado_civil_união estável formal,estado_civil_união estável informal,
#estado_civil_viúvo

e1 = features[['grau_escolaridade_fund. completo']].values
e2 = features[['grau_escolaridade_fund. incompleto']].values
e3 = features[['grau_escolaridade_médio completo']].values
e4 = features[['grau_escolaridade_médio incompleto']].values
e5 = features[['grau_escolaridade_pós-graduação']].values
e6 = features[['grau_escolaridade_superior completo']].values
e7 = features[['grau_escolaridade_superior incompleto']].values
s1 = features[['estado_civil_casado']].values
s2 = features[['estado_civil_divorciado']].values
s3 = features[['estado_civil_separado']].values
s4 = features[['estado_civil_solteiro']].values
s5 = features[['estado_civil_união estável formal']].values
s6 = features[['estado_civil_união estável informal']].values
s7 = features[['estado_civil_viúvo']].values

mod = smf.ols('e ~ zlog + w + xlog + e1 + e2 + e3 + e4 + e5 + e6 + e7 + s1 + s2 + s3 + s4 + s5 + s6 + s7', data=data)
resultados = mod.fit()
print(resultados.summary())

#R2 de 0.845 sem log
#r2 de 0.859 com log


teste = pd.read_csv("dados/escore_teste.csv")

numericas = teste.select_dtypes('number')
categoricas = teste.select_dtypes('object')

categoricas = pd.get_dummies(categoricas[categoricas.columns])

features2 = pd.concat([numericas, categoricas], axis=1)

features2.to_csv('features2.csv')


z = features2['salario'].values
x = features2[['experiência']].values
w = features2[['idade']].values
#e = features2[['escore']].values
xlog = np.log(x + 1)
zlog = np.log(z + 1)

e1 = features2[['grau_escolaridade_fund. completo']].values
e2 = features2[['grau_escolaridade_fund. incompleto']].values
e3 = features2[['grau_escolaridade_médio completo']].values
e4 = features2[['grau_escolaridade_médio incompleto']].values
e5 = features2[['grau_escolaridade_pós-graduação']].values
e6 = features2[['grau_escolaridade_superior completo']].values
e7 = features2[['grau_escolaridade_superior incompleto']].values
s1 = features2[['estado_civil_casado']].values
s2 = features2[['estado_civil_divorciado']].values
s3 = features2[['estado_civil_separado']].values
s4 = features2[['estado_civil_solteiro']].values
s5 = features2[['estado_civil_união estável formal']].values
s6 = features2[['estado_civil_união estável informal']].values
s7 = features2[['estado_civil_viúvo']].values

print(features2)
print(resultados.params)
#kk = features2[['salario', 'idade', 'experiência', 'grau_escolaridade_fund. completo', 'grau_escolaridade_fund. incompleto', 'grau_escolaridade_médio completo', 'grau_escolaridade_médio incompleto', 'grau_escolaridade_pós-graduação', 'grau_escolaridade_superior completo', 'grau_escolaridade_superior incompleto', 'estado_civil_casado', 'estado_civil_divorciado', 'estado_civil_separado', 'estado_civil_solteiro', 'estado_civil_união estável formal', 'estado_civil_união estável informal', 'estado_civil_viúvo']]
print("cuuuuuuuuuuuuuuuuuuuuuu")
#kk = pd.DataFrame(features2.data)
#print(kk)
#predito = resultados.get_prediction(kk)
predito = resultados.predict(features2)


#escorepredito = resultados.predict(resultados.params)
#print(escorepredito)
'''predito = resultados.predict([['e1']])
print(predito) #nao ta me dando o predito que eu quero
#print(resultados.summary())

predito.to_csv('predito2.csv')'''