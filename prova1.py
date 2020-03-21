import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

plt.rcParams['figure.figsize'] = [8,6]

dados2 = pd.read_csv("dados/escore_treino.csv")
print(dados2.head(25))
#print("\n", dados.describe(include='all')) #pra ver se nao tem missing data
print("\n", dados2.describe())
dados = dados2.dropna()
print("\n", dados.describe())
print(dados2.shape)
print(dados.shape) #elimina 52 observações :o
#sns.pairplot(dados)
'''plt.scatter(dados.idade, dados.escore)
plt.show()
plt.scatter(dados.experiência, dados.escore)
plt.show()'''
ohe = OneHotEncoder(drop='first')
colors = ohe.fit_transform(dados[['grau_escolaridade']])
print(ohe.categories_)
print(dados.grau_escolaridade)
print(colors.todense()) 

# ['fund. completo', 'fund. incompleto', 'médio completo', 'médio incompleto', 
# 'pós-graduação', 'superior completo','superior incompleto'],
z = dados['salario'].values
x = dados[['experiência']].values
y = dados['escore'].values

xlog = np.log(x + 1)
zlog = np.log(z + 1)

reg = LinearRegression()
reg.fit(xlog, y)
y_pred = reg.predict(xlog)

zlog.reshape(-1, 1)
reg.fit(zlog, y)
y_pred2 = reg.predict(zlog)


print(r2_score(y, y_pred))
print(r2_score(y, y_pred2))