import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\prana\Downloads\Expedia\predicting_hotel_availability-main\train.csv')

df.isnull().sum()

df.shape



df['reviews_per_month']=df['reviews_per_month'].fillna(0)

df.isnull().sum()

f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='yearly_availability', data=df)
plt.title('Availability vs Non Availability')
plt.xlabel('Class (1==Availability)')

df_no=df[df['yearly_availability']==0].shape
print(df_no)

df_yes=df[df['yearly_availability']==1].shape
print(df_yes)

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(),annot=True)

df.describe()

plt.figure(figsize=(12,8))
plt.boxplot(df['reviews_per_month'])
plt.ylim(0,10)

df[df['reviews_per_month']>5].sort_values(by='reviews_per_month')

plt.figure(figsize=(12,8))
plt.boxplot(df['owned_hotels'])
plt.ylim(0, 15)

plt.figure(figsize=(12,8))
plt.boxplot(df['minimum_nights'])
plt.ylim(0, 999)

df[df['minimum_nights']>300].sort_values(by='minimum_nights')

plt.figure(figsize=(12,8))
plt.boxplot(df['cost'])
#plt.ylim(0, 999)
df[df['cost']>200].sort_values(by='cost')

df=df.drop(index=[2309,1388,2402,2184,781])

df=df.drop(['id','owner_id'],axis=1)

df=pd.get_dummies(df,columns=['accommodation_type','region'])

df.columns=['Latitude','Longitude','Cost','Minimum_nights','Number_of_reviews','Reviews_per_month','Owned_hotels',
           'yearly_availability','Entire_home/apt','Private_room','Shared_room','Mahattan','Bronx','Brooklyn','Queens','Staten_Island']

df

df.shape

# Top 20 Features

corr_df_viz = df.corr()
corr_df_viz['feature'] = corr_df_viz.index

plt.figure(figsize=(10,6))
# make barplot
sns.barplot(x='feature',
            y="yearly_availability", 
            data=corr_df_viz, 
            order=corr_df_viz.sort_values('yearly_availability', ascending = False).feature)
# set labels
plt.xlabel("Feature", size=15)
plt.ylabel("Correlation between Yearly Availability", size=15)
plt.title("Top 20 Features", size=18)
plt.tight_layout()
plt.xticks(rotation=80)

# Model

X=df.drop('yearly_availability',axis=1)
y=df['yearly_availability']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=101)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=800)
rfc.fit(X_train,y_train)

rfc_predictions = rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid={'C':[0.1,1,10,100,1000],'gamma':[1.0,0.1,0.01,0.001,0.0001]}

grid=GridSearchCV(SVC(),param_grid,verbose=3)

svc_model=SVC(C=1000, gamma=0.0001)

grid.fit(X_train,y_train)

grid.best_estimator_

svc_model.fit(X_train,y_train)

svc_predictions=svc_model.predict(X_test)

import xgboost
from xgboost import XGBClassifier

params={
    'learning_rate':[0.05,0.10,0.15,0.20,0.25,0.30],
    'max_depth':[3,4,5,6,7,8,10,11,12,13,14,15],
    'min_child_weight':[1,3,5,7,9],
    'gamma':[0.2,0.3,0.4,0.5,0.6,0.7,0.8],
    'colsample_bytree':[0.3,0.4,0.5,0.6]}

classifier=XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5)

random_search.fit(X_train,y_train,eval_metric='rmse',verbose=3)

random_search.best_params_

xgb_model=XGBClassifier(min_child_weight= 7,
 max_depth=7,
 learning_rate= 0.15,
 gamma= 0.2,
 colsample_bytree= 0.4)
xgb_model.fit(X_train,y_train,eval_metric='rmse')
xgb_predictions=xgb_model.predict(X_test)

from sklearn.naive_bayes import GaussianNB
nm_model=GaussianNB()

nm_model.fit(X_train,y_train)

nm_predictions=nm_model.predict(X_test)

print('XGB Classifier')
print(confusion_matrix(y_test,xgb_predictions))
'\n'
print(classification_report(y_test,xgb_predictions))
print('-'*100,'\n')
print('Support Vector Classifier')
print(confusion_matrix(y_test,svc_predictions))
'\n'
print(classification_report(y_test,svc_predictions))
print('-'*100,'\n')
print('Random Forest Classifier')
print(confusion_matrix(y_test,rfc_predictions))
'\n'
print(classification_report(y_test,rfc_predictions))
print('-'*100,'\n')
('-'*100)
print('Naive Bayes Classifier')
print(confusion_matrix(y_test,nm_predictions))
'\n'
print(classification_report(y_test,nm_predictions))



# Test Data

df1=pd.read_csv(r'C:\Users\prana\Downloads\Expedia\predicting_hotel_availability-main\test.csv')

df1.isnull().sum()

df1['reviews_per_month']=df1['reviews_per_month'].fillna(0)

df1.isnull().sum()

df1

df2=df1.drop(['id','owner_id'],axis=1)

df2=pd.get_dummies(df2,columns=['accommodation_type','region'])

df2.head(3)

df2.columns=['Latitude','Longitude','Cost','Minimum_nights','Number_of_reviews','Reviews_per_month','Owned_hotels',
           'Entire_home/apt','Private_room','Shared_room','Mahattan','Bronx','Brooklyn','Queens','Staten_Island']

X_test=df2

df1

yearly_availability=rfc.predict(X_test)

Final_dataset=pd.DataFrame()

Final_dataset['ID']=df1['id']
Final_dataset['yearly_availabilty']=yearly_availability

Final_dataset.head(5)
