from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score

from data_process import make_data
import numpy as np

import tqdm
import pickle
import yaml

#configファイルの読み込み
with open("config.yaml") as f:
    config=yaml.safe_load(f)

#入力するデータファイルのパスを指定
data_path=config["data_path"]

label_dict=config["label_dict"]

#入力データを形態要素解析してシャッフルしたのち、ラベルと入力に分離させる
data,label=make_data(data_path,config["sentence_index"],config["label_index"],label_dict)

#bag of words
cvt=CountVectorizer(analyzer=lambda x:x)
cvt.fit(data)

bow_vec=cvt.transform(data)

bow_X=np.array(bow_vec.toarray())


"""
#tfidf
tfidf=TfidfVectorizer(analyzer=lambda x:x)
tfidf.fit(arasuji_all)
tfidf_vec=tfidf.transform(arasuji_all)
bow_X=np.array(tfidf_vec.toarray())
print(len(bow_X),bow_X[20])
"""

"""
#分散表現
model=Word2Vec(arasuji_all,window=4,min_count=1)

dis_X=[]

for x in arasuji_all:
    disx=[]
    for word in x:
        disx.append(model[word])
    disx=np.mean(np.array(disx),axis=0)
    #disx=np.sum(disx,axis=0)/len(x)
    dis_X.append(disx)

dis_X=np.array(dis_X)
"""

print("data num:",len(label))

#k-foldの設定
kf=KFold(n_splits=config["k-fold_splits"],shuffle=True)
x_vec,y_vec=bow_X,label
accuracy_all=[]
accuracy_max=0

for train_index,test_index in tqdm.tqdm(kf.split(x_vec,y_vec)):
    X_train=np.array([x_vec[i] for i in train_index])
    y_train=np.array([y_vec[i] for i in train_index])
    X_test=np.array([x_vec[i] for i in test_index])
    y_test=np.array([y_vec[i] for i in test_index])
    clf=SVC(kernel='linear',probability=True)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    y_test=np.array([int(i) for i in y_test])
    y_pred=np.array([int(i) for i in y_pred])
    # print(y_test)
    # print(y_pred)
    accuracy_all.append(accuracy_score(y_test,y_pred))
    #accuracy_all.append(f1_score(y_test,y_pred))
    if accuracy_all[-1]>accuracy_max:
        accuracy_max=max(accuracy_all)
        #精度が記録更新したらモデルを保存
        with open(config["save_path"],mode="wb") as fp:
            pickle.dump(clf,fp)

accuracy_all=np.array(accuracy_all)
print(accuracy_all)
print(min(accuracy_all),np.mean(accuracy_all),accuracy_max)

"""
#学習器
clf=SVC(kernel='linear',probability=True)
X_vec,y_vec=bow_X,label
#X_vec,y_vec=dis_X,label
X_train,X_test,y_train,y_test=train_test_split(X_vec,y_vec,test_size=0.2)
clf.fit(X_train,y_train)

#予測
y_pred=clf.predict(X_test)
print("label:",y_test)
print("pred:",y_pred)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


#モデルの保存
with open("model.pickle",mode="wb") as fp:
    pickle.dump(clf,fp)
"""