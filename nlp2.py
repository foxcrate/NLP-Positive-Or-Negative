import os
from os.path import isfile
import sklearn
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def load_files(path):
    all_files = []
    in_file = []
    files = os.listdir(path)
    for file in files:
        if '.txt' in file:
            all_files.append(os.path.join(path, file))
    for f in all_files:
        f = open(f, "r")
        if f.mode == 'r':
            in_file.append(f.read())
    f.close()
    return in_file


def tfidf(x, y):
    vectorizer = TfidfVectorizer(stop_words='english', analyzer= 'word' , lowercase=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state = 42,shuffle = False)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test,vectorizer


def label(docs, label):
    list_labels = dict.fromkeys(docs,label)
    return list_labels

def concat_data(pos,neg):
    dic_all = {}
    shuffled_dic = {}
    for d in [pos, neg]:
        dic_all.update(d)
    keys =  list(dic_all.keys())
    random.shuffle(keys)
    for temp_key in keys:
        shuffled_dic[temp_key] = dic_all[temp_key]
    y = list(shuffled_dic.values())
    return keys, y
    

def classifier(X_train, X_test, y_train, y_test):
    LogReg = LogisticRegression(random_state=0,solver='lbfgs',C=1e7)
    LogReg.fit(X_train, y_train)
    y_pred = LogReg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return LogReg,acc

def predict(vectorizer, log_reg, test ):
    input_pred = []
    input_pred.append(test)
    x = vectorizer.transform(input_pred)
    y_pred = log_reg.predict(x)
    return y_pred

def output():
    pos_data = load_files('pos')
    neg_data = load_files('neg')
    dictionary_pos = label(pos_data, 1)
    dictionary_neg = label(neg_data, 0)
    x , y = concat_data(dictionary_pos,dictionary_neg)
    X_train, X_test, y_train, y_test, vectorizer = tfidf(x, y)
    log_reg , accuracy = classifier(X_train, X_test, y_train, y_test)
    print(accuracy)
    test = input('enter your review ')
    pred_input = predict(vectorizer , log_reg , test)
    if pred_input == 1:
        print('Postive Review')
    else:
        print('Negative Review')


output()







