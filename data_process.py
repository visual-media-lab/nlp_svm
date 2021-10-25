import numpy as np
import MeCab
import random
import csv
import mojimoji
import yaml

#このサイトから取ってきた↓
#http://tyamagu2.xyz/articles/ja_text_classification/
class WordDividor:
    with open("config.yaml") as f:
        config=yaml.safe_load(f)
    INDEX_CATEGORY = 0
    INDEX_ROOT_FORM = 6
    #使用する品詞を変えたい場合はTARGET_CATEGORIESを変更する
    TARGET_CATEGORIES = config["TARGET_CATEGORIES"]
    EXCEPT_CATEGORIES = config["EXCEPT_CATEGORIES"]

    def __init__(self, dictionary="mecabrc"):
        self.dictionary = dictionary
        self.tagger = MeCab.Tagger(self.dictionary)

    def extract_words(self, text):
        if not text:
            return []
        words = []
        node = self.tagger.parseToNode(text)
        while node:
            features = node.feature.split(',')
            if features[1] in self.EXCEPT_CATEGORIES:
                node = node.next
                continue
            if features[self.INDEX_CATEGORY] in self.TARGET_CATEGORIES:
                if features[self.INDEX_ROOT_FORM] == "*":
                    words.append(node.surface)
                else:
                    # prefer root form
                    words.append(features[self.INDEX_ROOT_FORM])
            node = node.next
        return words

#CSVファイルから文章を取り出す
#sentence_index: 何列目に文章があるか
#label_index: 引っ張りたいラベルが何列目か
#label_dict: 各ラベルに対して割り振る番号を設定
def get_sentence(path,sentence_index,label_index,label_dict):
    #CSVの読み込み
    csv_file=open(path,"r",errors="",newline="")
    rawdata_L=csv.reader(csv_file,delimiter=",",doublequote=True,lineterminator="\r\n",quotechar='"',skipinitialspace=True)
    rawdata_L=list(rawdata_L)
    

    #データとラベルに分離
    L=[]
    label=[]
    for i in rawdata_L:
        l=i[label_index].split(",")
        sentence=i[sentence_index]

        #ラベルに半角文字と全角文字が混在する場合はコメントアウトを解除すること
        #l=[mojimoji.han_to_zen(i) for i in l]

        #もともとラベルがない場合は無視する
        if len(l)==0:
            continue

        if label_dict is not None:
            l="".join(map(str,[label_dict[i] for i in l if i in label_dict]))

        if "その他" not in l and len(sentence)>0:
            L.append(mojimoji.han_to_zen(sentence))#ここで半角文字はすべて全角文字にしている
            label.append(l)

    return L,label

def make_data(filename,sentence_index,label_index,label_dict):
    #文章とラベルに分離
    L,label=get_sentence(filename,sentence_index,label_index,label_dict)
    
    #文章を単語に分割
    wd=WordDividor()
    L=[wd.extract_words(i) for i in L]

    #単語数でフィルタリング
    words_len=[len(i) for i in L]
    len_std=np.std(np.array(words_len))
    len_ave=np.mean(np.array(words_len))
    T=[50+10*((i-len_ave)/len_std) for i in words_len]
    L=[L[i] for i in range(len(T)) if T[i]>=40 and T[i]<=60]
    label=[label[i] for i in range(len(T)) if T[i]>=40 and T[i]<=60]

    #データをシャッフルする
    x=[i for i in range(len(L))]
    random.shuffle(x)
    L_out=[L[i] for i in x]
    label_out=[label[i] for i in x]

    return L_out,np.array(label_out)