# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# データセット読み込み
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 目標値
df['target'] = iris.target

# 目標値を数字から花の名前に変更
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'

# 予測モデル構築 すべての行の、0列目sepal lengthと2列目petal lengthのみxとして使用
x = iris.data[:, [0, 2]] 
y = iris.target

# ロジスティック回帰
clf = LogisticRegression()
clf.fit(x, y)


# サイドバー（入力画面）
st.sidebar.header('Input Features')

sepalValue = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)

# メインパネル
st.title('Iris Classifier')
st.write('## Input Value')

# インプットデータ（1行のデータフレーム）
value_df = pd.DataFrame([],columns=['data','sepal length (cm)','petal length (cm)'])
record = pd.Series(['data',sepalValue, petalValue], index=value_df.columns)
value_df = pd.concat([value_df, pd.DataFrame({'data': ['data'], 'sepal length (cm)': [sepalValue], 'petal length (cm)': [petalValue]})], ignore_index=True)
value_df.set_index('data', inplace=True)

# 入力値の値の表示
st.write(value_df)

# 予測値のデータフレーム　出力された値の表示
pred_probs = clf.predict_proba(value_df)
# 行をprobability、列をアヤメの名称とする予測値のデータフレーム
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'],index=['probability'])

st.write('## Prediction') # アプリに表示
st.write(pred_df)

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist() # 予測値最大のものの名前
st.write('## Result')
st.write('このアイリスはきっと',str(name[0]),'です!')

### [補足] Web アプリケーションの公開(★★)
# 1.GitHub と Streamlit sharing の登録と連携
# 2.Streamlit で作成したファイルをローカルレポジトリへ加える
# 3.GitHub 上のリモートリポジトリへプッシュ
# 4.デプロイ

