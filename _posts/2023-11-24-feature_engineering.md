---
use_math : True
---
# 1.피쳐엔지니어링
## 1-1. 피쳐(feature)
데이터모델(특히, 인공지능)에서 예측을 수행하는 데 사용되는 입력변수를 의미한다. <br>
통계학에서는 독립변수라고 한다. <br>
<br>
상관관계: 독립변수<->종속변수<br>
인과관계: 독립변수->종속변수<br>
<br>
인과관계가 되기 위한 조건<br>
1. x, y가 상관관계<br>
2. 시간적 선후관계（x가 먼저)<br>
3. 비허위적 관계(non-spurious)<br>
<br>
💠피쳐의 유형<br>
-속성에 따라 범주형(범주나 순위가 있는 변수), 수치형(수치로 표현되는 변수)로 나뉜다.
<br>
<br>
인과관계에 따라 독립변수(다른 변수에 영향을 받지 않고 종속변수에 영향을 주는 변수), 종속변수(독립 변수로부터 영향을 받는 변수)로 나뉜다.
<br>
<br>
머신러닝에서의 입력: 변수(Feature), 속성(Attribute), 예측변수(Predictor), 차원(Dimension), 관측치(Observation), 독립변수(Independent Variable)<br>
머신러닝에서의 출력: 라벨(Label), 클래스(Class), 목푯값(Target), 반응(Response), 종속변수(Dependent Variable)<br>
<br>

## 1-2. 피쳐 엔지니어링(Feature Engineering)
피쳐 엔지니어링은 머신러닝 알고리즘의 성능을 향상시키기 위하여 데이터에 대한 도메인 지식을 활용하여 변수를 조합하거나 새로운 변수를 만드는 과정을 말한다.<br>
피쳐 추출, 피쳐 선택 모두 피쳐의 개수를 줄이는 방법이다.<br>
<br>
<br>
💠피쳐추출(feature extraction): 새로운 피쳐를 만드는 방법<br>
-피쳐들 사이에 내재한 특성이나 관계를 분석하여 이들을 잘 표현할 수 있는 새로운 선형 혹은 비선형 결합 변수를 만들어 데이터를 줄이는 방법<br>
-고차원의 원본 피쳐 공간을 저차원의 새로운 피쳐 공간으로 투영<br>
-PCA(주성분 분석), LDA(선형 판별 분석) 등<br>

💠피쳐 선택(feature selection): 피쳐의 수를 줄이는 방법<br>
-피쳐 중 타겟에 가장 관련성이 높은 피쳐만을 선정하여 피쳐의 수를 줄이는 방법<br>
-관련없거나 중복되는 피쳐들을 필터링하고 간결한 subset을 생성<br>
-모델 단순화, 훈련 시간 축소, 차원의 저주 방지, 과적합(Over-fitting)을 줄여 일반화해주는 장점이 있음<br>
-Filter, Wrapper, Embedded 메서드<br>
![png](https://drive.google.com/uc?id=1gjZXN4-EPa_tRuIkgRismbiFpfs9ODiN)
![png](https://drive.google.com/uc?id=1m9NRuVkNnJKL0gjt7lHR8ABRJ49uD0u6)<br>
사진을 보면 Feature Engineering의 방법이 엄청나게 많은데, 이는 dominant한 방법이 없다는 것을 뜻한다.<br>
# 2. 피쳐 추출
## 2-1.피쳐 추출(Feature Extraction)<br>
변수들 사이에 내재한 특성이나 관계를 분석하여 이들을 잘 표현할 수 있는 새로운 선형 혹은 비선형 결합 변수를 만들어 데이터를 줄이는 방법<br>
<br>
주성분 분석(Principal Component Analysis, PCA)<br>
-변수들의 공분산 행렬이나 상관행렬을 이용<br>
-원래 데이터 특징을 잘 설명해주는 성분을 추출하기 이하여 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간으로 변환하는 기법<br>
-행의 수와 열의 수가 같은 정방행렬에서만 사용<br>
<br>
선형 판별 분석(Linear Discriminant Analysis, LDA)<br>
-데이터의 Target값 클래스끼리 최대한 분리할 수 있는 축을 찾음<br>
-특정 공간상에서 클래스 분리를 최대화하는 축을 찾기 위해 클래스 간 분산(between-class scatter)과 클래스 내부 분산(within-class scatter)의 비율을 최대화하는 방식으로 차원을 축소<br>
<br>
특이값 분해(Singular Value Decomposition)<br>
-M X N 차원의 행렬데이터에서 특이값을 추출하고 이를 통해 주어진 데이터 세트를 효과적으로 축약할 수 있는 기법<br>
<br>
요인 분석(Factor Analysis)<br>
-데이터 안에 관찰할 수 있는 잠재적인 변수(Latent Variable)가 존재한다고 가정<br>
-모형을 세운 뒤 관찰 가능한 데이터를 이용하여 해당 잠재 요인을 도출하고 데이터 안의 구조를 해석하는 기법<br>
-주로 사회과학이나 설문 조사 등에서 많이 활용<br>
<br>
독립 성분 분석(Independent Component Analysis)<br>
-주성분 분석과는 달리 다변량의 신호를 통계적으로 독립적인 하부성분으로 분리하여 차원을 축소하는 기법<br>
-독립 성분의 분포는 비정규 분포를 따르게 되는 차원축소 기법<br>
<br>
다차원 척도법(Multi-Dimensional Scaling)<br>
-개체들 사이의 유사성, 비유사성을 측정하여 2차원 또는 3차원 공간상에 점으로 표현하여 개체들 사이의 집단화를 시각적으로 표현하는 분석 방법<br>
<br>
## 2-2 주성분 분석
💠 주성분 분석(Principal Component Analysis)<br>
<br>
가장 널리 사용되는 차원(변수) 축소 기법 중 하나<br>
원 데이터의 분산(variance)을 최대한 보존하면서 서로 직교하는 새 기저(축)를 찾아, 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간으로 변환하는 기법<br>
PCA는 기존의 변수를 조합하여 서로 연관성이 없는 새로운 변수, 즉 주성분(principal component, PC)들을 만들어 냄<br>
주성분의 개수를 증가시킴에 따라 원 데이터의 분산의 보존수준이 높아짐<br>
<br>💠 PCA 절차
<br>
학습 데이터셋에서 분산이 최대인 축(axis)을 찾음<br>
첫번째 축과 직교(orthogonal)하면서 분산이 최대인 두 번째 축을 찾음<br>
첫 번째 축과 두 번째 축에 직교하고 분산을 최대한 보존하는 세 번째 축을 찾음<br>
1~3과 같은 방법으로 데이터셋의 차원(특성 수)만큼의 축을 찾음<br>
<br>
## 2-3 선형판별분석
💠 선형판별분석(Linear Discriminant Analysis, LDA)<br>
입력 데이터 세트를 저차원 공간으로 투영(projection)해 차원을 축소하는 기법<br>
데이터의 Target값 클래스끼리 최대한 분리할 수 있는 축을 찾음 → 지도 학습<br>
PCA는 Target값을 사용하지 않으므로 비지도 학습<br>
<br>
💠 LDA 절차<br>
특정 공간상에서 클래스 분리를 최대화하는 축을 찾기 위해 클래스 간 분산(between-class scatter)과 클래스 내부 분산(within-class scatter)의 비율을 최대화하는 방식으로 차원을 축소<br>
SVM 같은 다른 분류 알고리즘을 적용하기 전에 차원을 축소시키는 데 사용<br>
## 2-4. Scikit-Learn으로 PCA와 LDA 수행하기
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


    
![png](https://drive.google.com/uc?id=1pM6jbyCcKqXHYEXOCwTT3rzr-6lmAh6C)
    


# 3. 피쳐 선택 기법
종속변수 활용여부에 따라<br>
-Supervised: 종속변수를 활용하여 선택<br>
-Unsupervised: 독립변수들만을 이용해서 선택(종속변수인 y가 없음)<br>
<br>
선택 메커니즘에 따라<br>
-Filter: 통계적인 방법으로 선택<br>
-Wrapper: 모델을 활용하여 선택<br>
-Embedded: 모델 훈련 과정에서 자동으로 선택<br>
-Hybrid: Filter + Wrapper<br>
<br>
## 3-1. 필터기법(Filter Method)
모든 피쳐 배정 -> 최적의 subset 선택 -> 학습 알고리즘 -> 성능평가<br>
<br>
-데이터의 통계적 측정 방법을 사용하여 변수들의 상관관계를 알아냄<br>
-계산속도가 빠르고 변수 간 상관관계를 알아내는 데 적합하여 래퍼 기법(Wrapper Method)을 사용하기 전에 전처리하는 데 사용<br>
-특정 모델링 기법에 의존하지 않고 데이터의 통계적 특성부터 변수를 택하는 기법<br>
<br>
💠 필터기법의 종류<br>
<br>
-분산 기반 선택(Variance-based Selection): 분산이 낮은 변수를 제거하는 방법<br>
-정보 소득(Information Gain): 가장 정보 소득이 높은 속성을 선택하여 데이터를 더 잘 구분하게 되는 것<br>
-카이제곱 검정(Chi-Square Test): 카이제곱 분포$χ_2$에 기초한 통계적 방법으로 관찰된 빈도가 기대되는 빈도와 의미있게 다른지 여부를 검증하기 위해 사용되는 검증 방법<br>
-피셔 스코어(Fisher Score): 최대 가능성 방정식을 풀기 위해 통계에 사용되는 뉴턴(Newton)의 방법<br>
-상관계수(Correlation Coefficient): 두 변수 사이의 통계적 관계를 표현하기 위해 특정한 상관관계의 정도를 수치적으로 나타낸 계수<br>

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
    
💠 Scikit-Learn 제공 피쳐 선택 메서드<br>
<br>
-SelectKBest(): 고정된 k개의 피쳐 선택기<br>
-SelectPercentile(): 분위수 기반 선택기<br>
-SelectFpr(): False positive rate 기반 선택기<br>
-SelectFdr(): 추정된 False discovery rate 기반 선택기<br>
-SelectFwe(): familiy-wise error rate 기반 선택기<br>
-GenericUnivariateSelect(): 단변량 피쳐 선택기<br>
💠 Scikit-Learn 제공 피쳐 선택 기준<br>
<br>
f_classif: ANOVA F-value 분류<br>
mutual_info_classif: 상호정보량(mutual information) 분류<br>
chi2: 카이제곱 분류<br>
f_regression: F-value 회귀<br>
mutual_info_regression: 상호정보량(mutual information) 회귀<br>

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
    

## 3-2. 래퍼 기법(Wrapper Method)
예측 정확도 측면에서 가장 좋은 성능을 보이는 하위 집합을 선택하는 기법<br>
검색 가능한 방법으로 하위 집합을 반복해서 선택하여 테스트하는 것이므로 탐욕 알고리즘(Greedy Algorithm)에 속함<br>
반복하여 선택하는 방법으로 시간이 오래 걸리고 부분집합의 수가 기하급수적으로 늘어 과적합의 위험이 발생할 수 있음<br>
일반적으로 래퍼 방법은 필터 방법보다 예측 정확도가 높음<br>
<br>
💠 변수 선택을 위한 알고리즘<br>
-전진 선택법(Forward Selection): 모형을 가장 많이 향상시키는 변수를 하나씩 점진적으로 추가하는 방법<br>
-후진 제거법(Backward Elimination): 모두 포함된 상태에서 시작하여 가장 적은 영향을 주는 변수부터 하나씩 제거<br>
-단계적 방법(Stepwise Method): 전진선택과 후향제거의 결합/ 각 단계에서 최상의 속성을 선택하고 나머지 속성 중 최악의 속성을 제거하는 과정을 실행<br>
-의사결정트리<br>
<br>
💠 래퍼기법의 종류<br>
-RFE(Recursive Feature Elimination): SVM(Support Vector Machine)을 사용하여 재귀적으로 제거하는 방법/ 전진 선택, 후진 제거, 단계적 방법 사용<br>
-SFS(Sequential Feature Selection): 그리디 알고리즘(Greedy Algorithm)으로 빈 부분 집합에서 특성 변수를 하나씩 추가하는 방법 /전진 선택, 후진 제거 사용<br>

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
    

## 3-3. 임베디드 기법(Embedded Method)
임베디드 기법은 모델의 정확도에 기여하는 변수를 학습함<Br>
SelectFromModel<br>
->의사결정나무 기반 알고리즘에서 변수를 선택하는 기법<br>
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


