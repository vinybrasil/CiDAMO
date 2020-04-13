import os
import pandas as pd 

def ler_csv(file, PASTA):
    print("Lendo ", file)
    if file.endswith(".csv"):
        DIRECTORIO = os.path.join(PASTA, file)
        data = pd.read_csv(DIRECTORIO)
        NEW_DF = pd.DataFrame(data)
        return NEW_DF
    else:
        pass 

if __name__ == '__main__':

    PRIMARY_DF = pd.DataFrame().rename_axis("Id", index=True)
    PASTA = "101_restaurantes"
    for file in os.listdir(PASTA):
        NEW_DF = ler_csv(file, PASTA)
        PRIMARY_DF = PRIMARY_DF.append(NEW_DF, ignore_index=True)
        print(PRIMARY_DF.shape)
    PRIMARY_DF.to_csv('final.csv')