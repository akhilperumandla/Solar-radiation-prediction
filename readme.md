import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegre
df= pd.read_csv('SolarPrediction.csv')
df
df= pd.read_csv('SolarPrediction.csv')
df
df=df.drop(['Data','Time','TimeSunRise','TimeSunSet'],axis=1)
df
figure= plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
figure= plt.figure(figsize=(20,10))
sns.pairplot(df)
X= df.drop('Radiation',axis=1)
y=df['Radiation']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
ada_reg= AdaBoostRegressor(RandomForestRegressor())
ada_reg.fit(X_train,y_train)
ada_reg.score(X_test,y_test)
y_pred=ada_reg.predict(X_test)
print(r2_score(y_test, y_pred))
