import requests
from bs4 import BeautifulSoup
import re
import pandas as pd 
import pickle




def get_data(url):
    req = requests.get(url)
    content = req.content
    print("deu boa")
    soup = BeautifulSoup(content, 'html.parser')
    return soup

def get_reviews(url):
    reviews = []
    respostas = []
    for item in url.findAll("div", {"class":"mgrRspnInline"}):
        for resposta in item.find_all("p", {"class":"partial_entry"}):
            respostas.append(resposta)
    for review in url.find_all("p", {"class":"partial_entry"}):
        #print(review)
        if review not in respostas:
            reviews.append(review)
    return reviews


def get_notas(url, i):
    notas = []
    notas_provisorio = []

    for e in url.findAll("div", {"id":"taplc_location_reviews_list_resp_rr_resp_0"}):
        #for el in e.find_all("span"):
        for el in e.find_all("span"):
            notas_provisorio.append(el)


    for element in url.find_all("span"):
        try:
            if "bubble_rating" in str(element.attrs):
                #if "style" not in str(element.attrs):
                #print(str(element.attrs))
                print(element)
                
                if element in notas_provisorio:
                #if element in url.findAll("div", {"id":"REVIEWS"}):
                    notas.append(str(element.attrs))
        except AttributeError:
            pass
    return notas

def limpar_notas(notas):
    notas_limpas = []
    for nota in notas:
        if "50" in nota:
            notas_limpas.append("50")
        elif "45" in nota:
            notas_limpas.append("45")
        elif "40" in nota:
            notas_limpas.append("40")
        elif "35" in nota:
            notas_limpas.append("35")
        elif "30" in nota:
            notas_limpas.append("30")
        elif "25" in nota:
            notas_limpas.append("25")
        elif "20" in nota:
            notas_limpas.append("20")
        elif "15" in nota:
            notas_limpas.append("15")
        elif "10" in nota:
            notas_limpas.append("10")
        elif "05" in nota:
            notas_limpas.append("05")
        elif "00" in nota:
            notas_limpas.append("00")
        else:
            print("deu ruim aqui")
    return notas_limpas

def loop():
    
    #SITE = get_data("https://www.tripadvisor.com.br/Restaurant_Review-g303441-d16787688-Reviews-Pizza_Bis_Batel-Curitiba_State_of_Parana.html")
    SITE = get_data('https://www.tripadvisor.com.br/Restaurant_Review-g303441-d16787688-Reviews-Pizza_Bis_Batel-Curitiba_State_of_Parana.html')
    REVIEWS = get_reviews(SITE)
    NOTAS = get_notas(SITE, len(REVIEWS))
    NOTAS_LIMPAS = limpar_notas(NOTAS)
    df = pd.DataFrame(list(zip(NOTAS_LIMPAS, REVIEWS)), columns=['notas', 'reviews'])
    return df

def main():
    DF_INICIAL = pd.DataFrame(columns=['notas', 'reviews'])
    df = loop()
    
    DF_FINAL = DF_INICIAL.append(df)
    print(DF_FINAL.head)


    #ta com problema pra achar as notas certas

if __name__ == '__main__':
    main()