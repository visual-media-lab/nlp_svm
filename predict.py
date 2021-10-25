from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


from gensim.models.word2vec import Word2Vec
import numpy as np

import copy
import random
import pickle
from data_process import make_data,WordDividor
import yaml

with open("config.yaml") as f:
    config=yaml.safe_load(f)

ans=config["ans"]

label_dict=config["label_dict"]

#入力データが入っているファイル名
data_path=config["data_path"]

#入力データを形態要素解析してシャッフルしたのち、ラベルと入力に分離させる
data,label=make_data(data_path,
    config["sentence_index"],
    config["label_index"],label_dict)

#bag of words
cvt=CountVectorizer(analyzer=lambda x:x)
cvt.fit(data)

#モデルの読み込み
with open(config["save_path"], mode="rb") as fp:
    clf=pickle.load(fp)

wd=WordDividor()

while True:
    data_in=input("判別したい文章を入力してください(改行は必ず省いてください): ")
    data_in_all=[wd.extract_words(data_in)]
    print(data_in_all)

    #bag of words
    bow_vec=cvt.transform(data_in_all)
    bow_X=np.array(bow_vec.toarray())

    print("予測:",ans[int(clf.predict(bow_X)[0])])
    print()

"""
#学習器
clf=SVC(kernel='linear',probability=True)
X_vec,y_vec=bow_X,label
X_train,X_test,y_train,y_test=train_test_split(X_vec,y_vec,test_size=0.3)
clf.fit(X_train,y_train)

#予測
y_pred=clf.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))

#モデルの保存
with open("model.pickle",mode="wb") as fp:
    pickle.dump(clf,fp)
"""
