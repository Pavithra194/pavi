import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
titanic_data=pd.read_csv("C:/Users/user/Desktop/Titanic.csv")
titanic_data.head(10)
total_rows = titanic_data['PassengerId'].count
print (total_rows)
#analysis
sns.set(style="darkgrid")
ax = sns.countplot(x="Survived", data=titanic_data)
bx=sns.countplot(x="Survived",hue="Sex",data=titanic_data)
cx=sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist(bins=20,figsize=(10,5))
dx=sns.countplot(x="SibSp",data=titanic_data)
titanic_data.info()
#removing null valuesand cleaning
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap="viridis")
titanic_data["age"].plot.hist()
titanic_data.drop("Cabin",axis=1,inplace=True)
titanic_data.head(5)
titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)
titanic_data.isnull().sum()
sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
sex.head(5)
embarked=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embarked.head(5)
pclass=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
pclass.head(5)
titanic_data=pd.concat([titanic_data,sex,embarked,pclass],axis=1)
titanic_data.head(5)
titanic_data.drop(["Fare""Sex","PassengerId","Embarked","Name","Ticket"],axis=1,inplace=True)
titanic_data.head(5)
titanic_data.drop(["Pclass","Fare"],axis=1,inplace=True)
#train data
x=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]
#
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.3, random_state=1)
#
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import classification_report
classification_report(y_test,y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)





