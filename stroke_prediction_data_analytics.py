# -*- coding: utf-8 -*-
"""Stroke Prediction Data Analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11Qis1Oxm1Yc-iHIeYQxhBZ7pnnmSZARX
"""

import zipfile, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""Data Understanding"""

local_zip = '/content/Stroke Prediction Dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.info()

stroke.head()

stroke.describe()

stroke['bmi'] = stroke['bmi'].replace(np.NaN, 0)
y = (stroke.avg_glucose_level == 0).sum()
z = (stroke.bmi == 0).sum()

print("Nilai 0 di kolom avg_glucose_level: ", y)
print("Nilai 0 di kolom bmi: ", z)

stroke = stroke.loc[(stroke['bmi'] != 0)]
stroke

"""Outliers"""

sns.boxplot(x=stroke['age'])

sns.boxplot(x=stroke['avg_glucose_level'])

sns.boxplot(x=stroke['bmi'])

Q1 = stroke.quantile(0.25)
Q3 = stroke.quantile(0.75)
IQR=Q3-Q1
stroke=stroke[~((stroke<(Q1-1.5*IQR))|(stroke>(Q3+1.5*IQR))).any(axis=1)]

stroke.shape

numerical_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'work_type', 'Residence', 'smoking_status']

cat_features = stroke.select_dtypes(include='object').columns.to_list()

for col in cat_features:
  sns.catplot(x=col, y="age", kind="bar", dodge=False, height = 4, aspect = 3,  data=stroke, palette="Set3")
  plt.title("Rata-rata 'age' Relatif terhadap - {}".format(col))

sns.pairplot(stroke, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = stroke.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""Data Preparation"""

stroke = stroke.drop(["id"], axis=1)
stroke

categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for column in categorical_columns:
    dummies = pd.get_dummies(stroke[column], prefix=column)
    stroke = pd.concat([stroke, dummies], axis=1)
    stroke = stroke.drop(columns=column)


stroke['bmi'] = stroke['bmi'].replace(np.NaN, 0)

stroke

from sklearn.model_selection import train_test_split

x = stroke.drop(["age"], axis=1)
y = stroke["age"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

print(f'Total # of sample in whole dataset: {len(x)}')
print(f'Total # of sample in train dataset: {len(x_train)}')
print(f'Total # of sample in test dataset: {len(x_test)}')

from sklearn.preprocessing import StandardScaler
numerical_features = ['avg_glucose_level', 'bmi']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()

x_train[numerical_features].describe().round(4)

"""Modeling"""

models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""K-Nearest Neighbor"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(x_train), y_true=y_train)

"""Random Forest"""

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(x_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)

"""Boosting Algorithm"""

from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(x_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(x_train), y_true=y_train)

"""Evaluasi Model"""

x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3

mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = x_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)