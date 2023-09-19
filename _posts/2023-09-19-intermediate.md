```python
import numpy as np
import pandas as pd

```


```python
#데이터프레임의 변수와 메서드 보기
df = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[13,23,33]}, index=['1st','2nd','3rd'])
print(dir(df)[:20])
```

    ['A', 'B', 'C', 'T', '_AXIS_LEN', '_AXIS_ORDERS', '_AXIS_TO_AXIS_NUMBER', '_HANDLED_TYPES', '__abs__', '__add__', '__and__', '__annotations__', '__array__', '__array_priority__', '__array_ufunc__', '__bool__', '__class__', '__contains__', '__copy__', '__dataframe__']
    


```python
#데이터프레임의 열
df = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[13,23,33]}, index=['1st','2nd','3rd'])
print(df.B)
```

    1st    12
    2nd    22
    3rd    32
    Name: B, dtype: int64
    


```python
#데이터프레임의 T(transpose)
df = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[13,23,33]},index=['1st','2nd','3rd'])
print(df.A)
```

    1st    11
    2nd    21
    3rd    31
    Name: A, dtype: int64
    


```python
#전치행렬
print(df.T)
```

       1st  2nd  3rd
    A   11   21   31
    B   12   22   32
    C   13   23   33
    


```python
#사칙연산
df1 = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[13,23,33]}, index=['1st','2nd','3rd'])
df2 = pd.DataFrame({'A':[11,21,41], 'B':[12,22,42], 'E':[14,24,44]}, index=['1st','2nd','4th'])
```


```python
#각 원소별 매칭되는 것만 더하기
print(df1+df2)
```

            A     B   C   E
    1st  22.0  24.0 NaN NaN
    2nd  42.0  44.0 NaN NaN
    3rd   NaN   NaN NaN NaN
    4th   NaN   NaN NaN NaN
    


```python
# 값이 없는 것은 0으로 대체하여 각 원소별 더하기 
print(df1.add(df2, fill_value=0))
```

            A     B     C     E
    1st  22.0  24.0  13.0  14.0
    2nd  42.0  44.0  23.0  24.0
    3rd  31.0  32.0  33.0   NaN
    4th  41.0  42.0   NaN  44.0
    


```python
 # 값이 없는 것은 1로 대체하여 각 원소별 곱하기
print(df1.mul(df2, fill_value=1))
```

             A      B     C     E
    1st  121.0  144.0  13.0  14.0
    2nd  441.0  484.0  23.0  24.0
    3rd   31.0   32.0  33.0   NaN
    4th   41.0   42.0   NaN  44.0
    


```python
#기존 데이터프레임
df = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[13,23,33]}, index=['1st','2nd','3rd'])
print(df)
```

          A   B   C
    1st  11  12  13
    2nd  21  22  23
    3rd  31  32  33
    


```python
#새로운 열 생성
print(df.assign(A_plus_B = df.A+df.B))
```

          A   B   C  A_plus_B
    1st  11  12  13        23
    2nd  21  22  23        43
    3rd  31  32  33        63
    


```python
#새로운 열 생성(callable)
print(df.assign(log_A = lambda x:np.log(x.A)))
```

          A   B   C     log_A
    1st  11  12  13  2.397895
    2nd  21  22  23  3.044522
    3rd  31  32  33  3.433987
    


```python
#열 삽입
df = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[13,23,33]}, index=['1st','2nd','3rd'])
df.insert(loc=0, column='D', value=[14,24,34])
df.insert(loc=2, column='E', value=5)
print(df)
```

          D   A  E   B   C
    1st  14  11  5  12  13
    2nd  24  21  5  22  23
    3rd  34  31  5  32  33
    


```python
#열 제거
df = df.drop(columns =['D', 'E'])
print(df)
```


```python
#열이름 변경
df = df.rename(columns = {'A':'aaa'})
print(df)
```

          D  aaa  E   B   C
    1st  14   11  5  12  13
    2nd  24   21  5  22  23
    3rd  34   31  5  32  33
    


```python
df = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[13,23,33]}, index=['1st','2nd','3rd'])
print(df)
```

          A   B   C
    1st  11  12  13
    2nd  21  22  23
    3rd  31  32  33
    


```python
#잘못된 수정 명령어
df['2nd','A'] = 201
print(df)
```

          A   B   C  (2nd, A)
    1st  11  12  13       201
    2nd  21  22  23       201
    3rd  31  32  33       201
    


```python
#위치 지정 수정(.loc)
df.loc['2nd','A'] = 222
print(df)
```

           A   B   C  (2nd, A)
    1st   11  12  13       201
    2nd  222  22  23       201
    3rd   31  32  33       201
    


```python
df.iloc[:,3] = 'NA'
print(df)
```

           A   B   C (2nd, A)
    1st   11  12  13       NA
    2nd  222  22  23       NA
    3rd   31  32  33       NA
    


```python
#NA값을 1111로 대체
df = df.replace('NA', 1111)
print(df)
```

           A   B   C  (2nd, A)
    1st   11  12  13      1111
    2nd  222  22  23      1111
    3rd   31  32  33      1111
    


```python
#B열의 32을 9999로 대체
df = df.replace({'B':32}, 9999)
print(df)
```

           A     B   C  (2nd, A)
    1st   11    12  13      1111
    2nd  222    22  23      1111
    3rd   31  9999  33      1111
    


```python
df = pd.DataFrame({'A':[11,21,31], 'B':[12,22,32], 'C':[33,32,31]}, index=['1st','2nd','3rd'])
print(df)
```

          A   B   C
    1st  11  12  33
    2nd  21  22  32
    3rd  31  32  31
    


```python
#A의 값 기준 역순으로 정렬
df = df.sort_values(by='A', ascending=False)
print(df)
```

      order   A   B   C
    2   3rd  31  32  31
    1   2nd  21  22  32
    0   1st  11  12  33
    


```python
#행 index 정렬
df = df.sort_index(axis=0)
print(df)
```

      order   A   B   C
    0   1st  11  12  33
    1   2nd  21  22  32
    2   3rd  31  32  31
    


```python
#열 기준, 평균순위, 역순
df_rank = df.rank(axis=0, method='average', ascending=False)
print(df_rank)
```

       order    A    B    C
    0    3.0  3.0  3.0  1.0
    1    2.0  2.0  2.0  2.0
    2    1.0  1.0  1.0  3.0
    


```python
df = pd.DataFrame({'order': ['1st','2nd','3rd'],'A':[11,21,31], 'B':[12,22,32], 'C':[33,32,31]})
print(df)
```

      order   A   B   C
    0   1st  11  12  33
    1   2nd  21  22  32
    2   3rd  31  32  31
    


```python
#melt:속성이 같은 것들을 같은 열로 만들어서 데이터프레임 형태로 만듦
df_melted = pd.melt(df, id_vars=['order'], value_vars=['A','B','C'], var_name='name', value_name='score')
print(df_melted)
```

      order name  score
    0   1st    A     11
    1   2nd    A     21
    2   3rd    A     31
    3   1st    B     12
    4   2nd    B     22
    5   3rd    B     32
    6   1st    C     33
    7   2nd    C     32
    8   3rd    C     31
    


```python
#데이터 개수를 구함
print(df.count(axis=0))
```

    order    3
    A        3
    B        3
    C        3
    dtype: int64
    


```python
df_numeric = df[['A','B','C']]
```


```python
#평균
#axis = 1:행별
df_numeric.mean(axis = 1)
```




    2    31.333333
    1    25.000000
    0    18.666667
    dtype: float64




```python
#최대값
#axis = 0:열별
df_numeric.max(axis = 0)
```




    A    31
    B    32
    C    33
    dtype: int64




```python
#표본분산
#ddof = 표본 자유도 반영
df_numeric.var(axis = 1,ddof = 1)
```




    2      0.333333
    1     37.000000
    0    154.333333
    dtype: float64




```python
#상관계수
df_numeric.corr()
```




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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#describe-descriptive statistics(기술통계량)의 줄임말
df = pd.DataFrame({'A':[11,21,31,41], 'B':[12,22,32,42], 'C':[13,23,33,43]}, index=['1st','2nd','3rd','4th'])
print(df.describe())
```

                   A          B          C
    count   4.000000   4.000000   4.000000
    mean   26.000000  27.000000  28.000000
    std    12.909944  12.909944  12.909944
    min    11.000000  12.000000  13.000000
    25%    18.500000  19.500000  20.500000
    50%    26.000000  27.000000  28.000000
    75%    33.500000  34.500000  35.500000
    max    41.000000  42.000000  43.000000
    


```python
#샘플링
print(df.sample(n=2))
```

          A   B   C
    2nd  21  22  23
    4th  41  42  43
    


```python
print(df.sample(frac=0.5))
```

          A   B   C
    3rd  31  32  33
    1st  11  12  13
    


```python
df = pd.DataFrame({'A':[11,21,31,None,31], 'B':[12,22,32,42,32], 'C':[13,None,33,43,33]}, index=['1st','2nd','3rd','4th','7th'])
print(df)
```

            A   B     C
    1st  11.0  12  13.0
    2nd  21.0  22   NaN
    3rd  31.0  32  33.0
    4th   NaN  42  43.0
    7th  31.0  32  33.0
    


```python
#중복데이터 제거
print(df.drop_duplicates())
```

            A   B     C
    1st  11.0  12  13.0
    2nd  21.0  22   NaN
    3rd  31.0  32  33.0
    4th   NaN  42  43.0
    


```python
#결측치 행 제거
#default는 행을 제거함=>df.dropna()또는 df.dropna(axis=0)를 쓰면 행 제거
#df.dropna(axis = 1)을 쓰면 결측값이 있는 열 제거
print(df.dropna())
```


```python
print(df)
```

            A   B     C
    1st  11.0  12  13.0
    2nd  21.0  22   NaN
    3rd  31.0  32  33.0
    4th   NaN  42  43.0
    7th  31.0  32  33.0
    


```python
print(df.fillna(axis = 1, method = 'ffill'))
```

            A     B     C
    1st  11.0  12.0  13.0
    2nd  21.0  22.0  22.0
    3rd  31.0  32.0  33.0
    4th   NaN  42.0  43.0
    7th  31.0  32.0  33.0
    


```python
print(df.fillna(df.mean()))
```

            A   B     C
    1st  11.0  12  13.0
    2nd  21.0  22  30.5
    3rd  31.0  32  33.0
    4th  23.5  42  43.0
    7th  31.0  32  33.0
    


```python
#조건에 부합하는 데이터 추출
#.loc[]보다 속도가 느림
df = pd.DataFrame({'abc':[1,4,7], 'bcd':[2,5,8], 'abd':[3,6,9]}, index=['1st','2nd','3rd'])
print(df)
```

         abc  bcd  abd
    1st    1    2    3
    2nd    4    5    6
    3rd    7    8    9
    


```python
#질의어로 선택
print(df.query('abc > 3'))
```

         abc  bcd  abd
    2nd    4    5    6
    3rd    7    8    9
    


```python
print(df.query('(abc > 3) & (abd < 9)'))
```

         abc  bcd  abd
    2nd    4    5    6
    


```python
#'@'로 외부값(함수) 참조 가능
abd_max = 9
print(df.query('(abc > 3) & (abd < @abd_max)'))
```

         abc  bcd  abd
    2nd    4    5    6
    


```python
df = pd.DataFrame({'scale':['small','large','small','large'], 'location':['east','east','south','south'], 'sales':[10,20,30,40]})
print(df)
```

       scale location  sales
    0  small     east     10
    1  large     east     20
    2  small    south     30
    3  large    south     40
    


```python
#'scale'별로 그룹을 나누어 'sales'의 합계를 구함
data_s = df.groupby(by='scale')['sales'].sum()
print(data_s)
print(type(data_s)) 
print(data_s.index)
print(data_s.values)
```

    scale
    large    60
    small    40
    Name: sales, dtype: int64
    <class 'pandas.core.series.Series'>
    Index(['large', 'small'], dtype='object', name='scale')
    [60 40]
    


```python
#'location','scale'별로 그룹을 나누어 'sales'의 평균을 구함
data_sl = df.groupby(by=['location', 'scale'])['sales'].mean()
print(data_sl)
print(type(data_sl))
print(data_sl.index)
print(data_sl.values)
```

    location  scale
    east      large    20.0
              small    10.0
    south     large    40.0
              small    30.0
    Name: sales, dtype: float64
    <class 'pandas.core.series.Series'>
    MultiIndex([( 'east', 'large'),
                ( 'east', 'small'),
                ('south', 'large'),
                ('south', 'small')],
               names=['location', 'scale'])
    [20. 10. 40. 30.]
    


```python
#'location','scale'로 그룹을 나누어 'sales'의 평균을 구하여 데이터프레임으로 변
data_sl = df.groupby(by=['location', 'scale'])[['sales']].mean()
print(data_sl)
print(type(data_sl))
print(data_sl.index)
print(data_sl.values)
```

                    sales
    location scale       
    east     large   20.0
             small   10.0
    south    large   40.0
             small   30.0
    <class 'pandas.core.frame.DataFrame'>
    MultiIndex([( 'east', 'large'),
                ( 'east', 'small'),
                ('south', 'large'),
                ('south', 'small')],
               names=['location', 'scale'])
    [[20.]
     [10.]
     [40.]
     [30.]]
    


```python
df = pd.DataFrame({'scale':['small','large','small','large'], 'location':['east','east','south','south'], 'sales':[10,20,30,40]})
print(df)
```

       scale location  sales
    0  small     east     10
    1  large     east     20
    2  small    south     30
    3  large    south     40
    


```python
#python의 내장함수인 map함수 사용
print(list(map(lambda x: x**2, df.sales)))
```

    [100, 400, 900, 1600]
    


```python
#Apply-객체(함수)를 반복하여 적용
print(df.sales.apply(lambda x: x**2))
```

    0     100
    1     400
    2     900
    3    1600
    Name: sales, dtype: int64
    


```python
df1 = pd.DataFrame({'id':['1st','2nd','3rd'], 'name': ['홍길동', '임꺽정', '김홍익']})
df2 = pd.DataFrame({'id':['2nd','3rd','4th'], 'address': ['서울', '강원도', '경기도']})
print(df1)
print(df2)
```

        id name
    0  1st  홍길동
    1  2nd  임꺽정
    2  3rd  김홍익
        id address
    0  2nd      서울
    1  3rd     강원도
    2  4th     경기도
    


```python
#concat:합치는 기능/axis=0은 행으로, axis=1은 열로 합침
concat_row = pd.concat([df1,df2], axis = 0)
concat_col = pd.concat([df1,df2], axis = 1)

print('행으로 합침: \n', concat_row)
print('열로 합침: \n', concat_col)
```

    행으로 합침: 
         id name address
    0  1st  홍길동     NaN
    1  2nd  임꺽정     NaN
    2  3rd  김홍익     NaN
    0  2nd  NaN      서울
    1  3rd  NaN     강원도
    2  4th  NaN     경기도
    열로 합침: 
         id name   id address
    0  1st  홍길동  2nd      서울
    1  2nd  임꺽정  3rd     강원도
    2  3rd  김홍익  4th     경기도
    


```python
#내부결합-두개의 테이블 키가 일치하는 데이터만 추출(how = 'inner')
#외부결합-두개의 테이블 키와 관련된 모든 데이터 추출(how = 'outer')
inner_join = pd.merge(df1, df2, on='id', how='inner')
outer_join = pd.merge(df1, df2, on='id', how='outer')
print('inner: \n', inner_join)
print('outer: \n', outer_join)
```

    inner: 
         id name address
    0  2nd  임꺽정      서울
    1  3rd  김홍익     강원도
    outer: 
         id name address
    0  1st  홍길동     NaN
    1  2nd  임꺽정      서울
    2  3rd  김홍익     강원도
    3  4th  NaN     경기도
    


```python
#좌 결합- 왼쪽 테이블 키와 일치하는 데이터 추출
#우 결합- 오른쪽 테이블 키와 일치하는 데이터 추출
left_join = pd.merge(df1, df2, on='id', how='left')
right_join = pd.merge(df1, df2, on='id', how='right')
print('left: \n', left_join)
print('right: \n', right_join)
```

    left: 
         id name address
    0  1st  홍길동     NaN
    1  2nd  임꺽정      서울
    2  3rd  김홍익     강원도
    right: 
         id name address
    0  2nd  임꺽정      서울
    1  3rd  김홍익     강원도
    2  4th  NaN     경기도
    
