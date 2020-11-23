#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

df=pd.read_csv('prediction_data.csv')


df.head()

df.shape

print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


##check missing values
df.isnull().sum()

df.describe()

final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]

final_dataset.head()

final_dataset['Current Year']=2020

final_dataset.head()

final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']

final_dataset.head()

final_dataset.drop(['Year'],axis=1,inplace=True)

final_dataset.head()

final_dataset=pd.get_dummies(final_dataset,drop_first=True)

final_dataset.head()

final_dataset=final_dataset.drop(['Current Year'],axis=1)

final_dataset.head()

final_dataset.corr()

import seaborn as sns
from matplotlib import pyplot as plt


fig=plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)

X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]

X.head()

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

import xgboost as xgb
# Using XGBoost
xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train,y_train)

classifier=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

predictions = xgb_classifier.predict(X_test)

predictions

# Feature Importance
xgb.plot_importance(xgb_classifier)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)

import pickle
# open a file, where you ant to store the data
file = open('XGBoost_model.pkl', 'wb')

# dump information to that file
pickle.dump(xgb_classifier, file)




