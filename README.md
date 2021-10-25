# nlp_svm
自然言語処理用のSVM学習器
# 使い方
## 最初にやること
mecabのインストールを最初に行ってください.  
インストール方法↓  
[ubuntu](https://qiita.com/ekzemplaro/items/c98c7f6698f130b55d53)  
[windows](https://qiita.com/menon/items/f041b7c46543f38f78f7) (Anacondaが推奨されていますがこのためにわざわざAnacondaを入れなくてもいいと思います)
## 学習の設定
`config.yaml`を開いて中身を編集してください.
## 学習
```bash
~$ pip3 install -r requirements.txt
~$ python3 train.py
```
## 推論
```bash
~$ pip3 install -r requirements.txt
~$ python3 predict.py
```

## 最後に
バグ等ありましたらissueまで.