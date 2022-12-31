# INTRODUCTION

 ###### Cereal is a popular breakfast food in the United States, consisting of processed grains that are often mixed with sweeteners and served with milk. It is a quick and convenient option that is widely available in supermarkets and consumed by people of all ages. The first cereals were developed in the late 19th century and were initially marketed as a health food. Today, there are numerous varieties of cereals available, ranging from traditional oats and corn flakes to more innovative flavors and formulations. Many cereals are fortified with vitamins and minerals and are often marketed as a nutritious choice for breakfast. In addition to being eaten as a breakfast food, cereal is also commonly used as an ingredient in baked goods and as a topping for desserts. 
 ---






```import pandas as pd```

```import numpy as np```

```from sklearn.model_selection import train_test_split```

```from sklearn.preprocessing import StandardScaler```

```from sklearn.linear_model import LinearRegression```

```from sklearn.ensemble import RandomForestRegressor```


```import math```

#read the dataset

```emp_data = pd.read_csv('C:/Users/ozann/OneDrive/Masaüstü/databank/data101  proje/cereal.csv', low_memory= False)```

```def copy_df(df):```


```df = df.copy()```
    
```return df```


    
```data = copy_df(emp_data)```


```data.info()```


```data.iloc[:,[0,15]]```

| name     |  rating  |
|100% Bran |68.402973 |
|Trix      |27.753301 |
|Wheaties  |51.592193 |
|Honey Gold|36.187559 |

---



```df.type.astype('category').cat.codes```

#there is no null value

```df['type'] = df['type'].replace({'C': 0, 'H': 1})```


```y = df['rating']```
```x = df.drop('rating', axis=1)```


# Splitting train and test
```x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=2)```

# Scale x
```scaler = StandardScaler()```
```scaler.fit(x_train)```
```x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)```
```x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)```

```x_train```

```x_test```

```models = {```
```"Linear Regression : LinearRegression(),```
```"Random Forest    " : RandomForestRegressor()}```
`for name, model in models.items():`
    `model.fit(x_train, y_train)`
    
```for name, model in models.items():```
```print(name + " R^2 Score: {:.5f}".format(model.score(x_test, y_test)))```


   
