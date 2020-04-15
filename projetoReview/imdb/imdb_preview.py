from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd 
import os
import re 

#pega o dataset daqui: https://github.com/kmeagle1515/IMBD_SENTIMENT

reviews_train = [] 
for line in open("full_train.txt", 'r', encoding='utf-8'):
    reviews_train.append(line.strip())

reviews_test = []
for line in open("full_test.txt", "r", encoding='utf-8'):
    reviews_test.append(line.strip())


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

cv  = CountVectorizer(binary=True) #If True, all non zero counts are set to 1. 
                                   #This is useful for discrete probabilistic models that model binary events 
                                   #rather than integer counts.
cv2 = cv.fit(reviews_train_clean)
X = cv2.transform(reviews_train_clean)
X_test = cv2.transform(reviews_test_clean)

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, Y_train, Y_val = train_test_split(X, target, train_size=0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, Y_train)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(Y_val, lr.predict(X_val))))



#=================melhor modelo===============================#
final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test)))

print(final_model.coef_)
#print(cv.get_feature_names())

#========as palavras que mais tem peso========================#
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
print(type(feature_to_coef))
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
#     ('excellent', 0.9288812418118644)
#     ('perfect', 0.7934641227980576)
#     ('great', 0.675040909917553)
#     ('amazing', 0.6160398142631545)
#     ('superb', 0.6063967799425831)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    
#     ('worst', -1.367978497228895)
#     ('waste', -1.1684451288279047)
#     ('awful', -1.0277001734353677)
#     ('poorly', -0.8748317895742782)
#     ('boring', -0.8587249740682945)