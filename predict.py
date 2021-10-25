from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

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