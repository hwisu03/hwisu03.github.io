```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# iris 데이터셋을 로드
iris = datasets.load_iris()

X = iris.data # iris 데이터셋의 피쳐들
y = iris.target # iris 데이터셋의 타겟
target_names = list(iris.target_names) # iris 데이터셋의 타겟 이름

print(f'{X.shape = }, {y.shape = }') # 150개 데이터, 4 features
print(f'{target_names = }')  
```

    X.shape = (150, 4), y.shape = (150,)
    target_names = ['setosa', 'versicolor', 'virginica']
    


```python
# PCA의 객체를 생성, 차원은 2차원으로 설정(현재는 4차원)
pca = PCA(n_components=2)

# PCA를 수행. PCA는 비지도 학습이므로 y값을 넣지 않음
pca_fitted = pca.fit(X)

print(f'{pca_fitted.components_ = }')  # 주성분 벡터
print(f'{pca_fitted.explained_variance_ratio_ = }') # 주성분 벡터의 설명할 수 있는 분산 비율

X_pca = pca_fitted.transform(X) # 주성분 벡터로 데이터를 변환
print(f'{X_pca.shape = }')  # 4차원 데이터가 2차원 데이터로 변환됨
```

    pca_fitted.components_ = array([[ 0.36138659, -0.08452251,  0.85667061,  0.3582892 ],
           [ 0.65658877,  0.73016143, -0.17337266, -0.07548102]])
    pca_fitted.explained_variance_ratio_ = array([0.92461872, 0.05306648])
    X_pca.shape = (150, 2)
    


```python
# LDA의 객체를 생성. 차원은 2차원으로 설정(현재는 4차원)
lda = LinearDiscriminantAnalysis(n_components=2)

# LDA를 수행. LDA는 지도학습이므로 타겟값이 필요
lda_fitted = lda.fit(X, y)

print(f'{lda_fitted.coef_=}') # LDA의 계수
print(f'{lda_fitted.explained_variance_ratio_=}') # LDA의 분산에 대한 설명력

X_lda = lda_fitted.transform(X)
print(f'{X_lda.shape = }')  # 4차원 데이터가 2차원 데이터로 변환됨
```

    lda_fitted.coef_=array([[  6.31475846,  12.13931718, -16.94642465, -20.77005459],
           [ -1.53119919,  -4.37604348,   4.69566531,   3.06258539],
           [ -4.78355927,  -7.7632737 ,  12.25075935,  17.7074692 ]])
    lda_fitted.explained_variance_ratio_=array([0.9912126, 0.0087874])
    X_lda.shape = (150, 2)
    


```python
# 시각화 하기
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Seaborn을 이용하기 위해 데이터프레임으로 변환
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_lda = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
y = pd.Series(y).replace({0:'setosa', 1:'versicolor', 2:'virginica'})

# subplot으로 시각화
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

sns.scatterplot(df_pca, x='PC1', y='PC2', hue=y, style=y, ax=ax[0], palette='Set1')
ax[0].set_title('PCA of IRIS dataset')

sns.scatterplot(df_lda, x='LD1', y='LD2', hue=y, style=y, ax=ax[1], palette='Set1')
ax[1].set_title('LDA of IRIS dataset')

plt.show()
```


    
![png](output_3_0.png)
    



```python
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

# iris 데이터셋을 로드
iris = datasets.load_iris()

X = iris.data # iris 데이터셋의 피쳐들
y = iris.target # iris 데이터셋의 타겟
X_names = iris.feature_names # iris 데이터셋의 피쳐 이름
y_names = iris.target_names # iris 데이터셋의 타겟 이름

# 분산이 0.2 이상인 피쳐들만 선택하도록 학습
sel = VarianceThreshold(threshold=0.2).fit(X)
print(f'{sel.variances_ = }') # 각 피쳐의 분산 확인

# 분산이 0.2 이상인 피쳐들만 선택 적용
X_selected = sel.transform(X) # 분산이 0.2 이상인 피쳐들만 선택
X_selected_names = [X_names[i] for i in sel.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')
```

    sel.variances_ = array([0.68112222, 0.18871289, 3.09550267, 0.57713289])
    X_selected_names = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
    X_selected[:5] = array([[5.1, 1.4, 0.2],
           [4.9, 1.4, 0.2],
           [4.7, 1.3, 0.2],
           [4.6, 1.5, 0.2],
           [5. , 1.4, 0.2]])
    


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression, chi2

# k개의 베스트 피쳐를 선택
sel_fc = SelectKBest(f_classif, k=2).fit(X, y)
print('f_classif: ')
print(f'{sel_fc.scores_ = }')
print(f'{sel_fc.pvalues_ = }')
print(f'{sel_fc.get_support() = }')
print('Selected features: ', [X_names[i] for i in sel_fc.get_support(indices=True)]) # 선택된 피쳐들의 이름

sel_fr = SelectKBest(f_regression, k=2).fit(X, y)
print('\nf_regression: ')
print(f'{sel_fr.scores_ = }')
print(f'{sel_fr.pvalues_ = }')
print(f'{sel_fr.get_support() = }')
print('Selected features: ', [X_names[i] for i in sel_fr.get_support(indices=True)]) # 선택된 피쳐들의 이름

sel_chi2 = SelectKBest(chi2, k=2).fit(X, y)
print('\nchi2: ')
print(f'{sel_chi2.scores_ = }')
print(f'{sel_chi2.pvalues_ = }')
print(f'{sel_chi2.get_support() = }')
print('Selected features: ', [X_names[i] for i in sel_chi2.get_support(indices=True)]) # 선택된 피쳐들의 이름
```

    f_classif: 
    sel_fc.scores_ = array([ 119.26450218,   49.16004009, 1180.16118225,  960.0071468 ])
    sel_fc.pvalues_ = array([1.66966919e-31, 4.49201713e-17, 2.85677661e-91, 4.16944584e-85])
    sel_fc.get_support() = array([False, False,  True,  True])
    Selected features:  ['petal length (cm)', 'petal width (cm)']
    
    f_regression: 
    sel_fr.scores_ = array([ 233.8389959 ,   32.93720748, 1341.93578461, 1592.82421036])
    sel_fr.pvalues_ = array([2.89047835e-32, 5.20156326e-08, 4.20187315e-76, 4.15531102e-81])
    sel_fr.get_support() = array([False, False,  True,  True])
    Selected features:  ['petal length (cm)', 'petal width (cm)']
    
    chi2: 
    sel_chi2.scores_ = array([ 10.81782088,   3.7107283 , 116.31261309,  67.0483602 ])
    sel_chi2.pvalues_ = array([4.47651499e-03, 1.56395980e-01, 5.53397228e-26, 2.75824965e-15])
    sel_chi2.get_support() = array([False, False,  True,  True])
    Selected features:  ['petal length (cm)', 'petal width (cm)']
    


```python
# RFE(Recursive Feature Elimination) 적용
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.svm import SVC, SVR

# iris 데이터셋 로드
X, y = load_iris(return_X_y=True)

# 분류기 SVC 객체 생성, 선형분류, 3개의 클래스 
svc = SVR(kernel="linear", C=3)

# RFE 객체 생성, 2개의 피쳐 선택, 1개씩 제거 
rfe = RFE(estimator=svc, n_features_to_select=2, step=1)
# RFE+CV(Cross Validation), 5개의 폴드, 1개씩 제거
rfe_cv = RFECV(estimator=svc, step=1, cv=5) 

# 데이터셋에 RFE 적용
rfe.fit(X, y)
print('RFE Rank: ', rfe.ranking_)

# rank가 1인 피쳐들만 선택
X_selected = rfe.transform(X) 
X_selected_names = [X_names[i] for i in rfe.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')

# 데이터셋에 RFECV 적용
rfe_cv.fit(X, y)
print('RFECV Rank: ', rfe_cv.ranking_)

# rank가 1인 피쳐들만 선택
X_selected = rfe_cv.transform(X) 
X_selected_names = [X_names[i] for i in rfe_cv.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')
```

    RFE Rank:  [2 3 1 1]
    X_selected_names = ['petal length (cm)', 'petal width (cm)']
    X_selected[:5] = array([[1.4, 0.2],
           [1.4, 0.2],
           [1.3, 0.2],
           [1.5, 0.2],
           [1.4, 0.2]])
    RFECV Rank:  [1 2 1 1]
    X_selected_names = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
    X_selected[:5] = array([[5.1, 1.4, 0.2],
           [4.9, 1.4, 0.2],
           [4.7, 1.3, 0.2],
           [4.6, 1.5, 0.2],
           [5. , 1.4, 0.2]])
    


```python
# SFS(Sequential Feature Selector) : 순차적으로 특성을 선택하는 방법

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# 데이터를 로드하고, 분류기를 초기화한 후 SFS를 적용
X, y = load_iris(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=2, direction='backward')

# SFS를 학습하고, 선택된 특성을 출력
sfs.fit(X, y)
print('SFS selected: ', sfs.get_support())

# 선택된 피쳐들만 선택
X_selected = sfs.transform(X) 
X_selected_names = [X_names[i] for i in sfs.get_support(indices=True)] # 선택된 피쳐들의 이름

print(f'{X_selected_names = }')
print(f'{X_selected[:5] = }')
```

    SFS selected:  [False False  True  True]
    X_selected_names = ['petal length (cm)', 'petal width (cm)']
    X_selected[:5] = array([[1.4, 0.2],
           [1.4, 0.2],
           [1.3, 0.2],
           [1.5, 0.2],
           [1.4, 0.2]])
    


```python
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.datasets import load_iris

# 데이터를 로드하고, 분류기를 초기화한 후 SFS를 적용
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
sfm = SelectFromModel(estimator=clf)

# 모형 구조 확인 및 출력을 pandas로 설정
sfm.set_output(transform='pandas')
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SelectFromModel(estimator=DecisionTreeClassifier())</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">SelectFromModel</label><div class="sk-toggleable__content"><pre>SelectFromModel(estimator=DecisionTreeClassifier())</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div></div></div></div></div></div>




```python
# 모형 학습
sfm.fit(X, y)
print('SFM threshold: ', sfm.threshold_)

# 선택된 피쳐들만 선택
X_selected = sfm.transform(X) 
X_selected.columns = [X_names[i] for i in sfm.get_support(indices=True)] # 선택된 피쳐들의 이름

X_selected.head()
```

    SFM threshold:  0.25
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>


