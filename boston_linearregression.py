import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
bh=datasets.load_boston()
bh.keys()
print(bh["DESCR"])
bh.feature_names
df=pd.DataFrame(data=bh.data,columns=bh.feature_names)
df["price"]=bh.target
df.head()
df["price"].plot.hist(bins=50)
sns.pairplot(df[['CRIM','RM','AGE','LSTAT','DIS','ZN','price']])
sns.heatmap(df[['CRIM','RM','AGE','LSTAT','DIS','ZN','price']].corr(),annot=True)
x=df[['CRIM','RM','AGE','LSTAT','DIS','ZN']]
y=df["price"]
x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.33, random_state=1)
x_train.info()
x_test.info()
from sklearn import linear_model
lm=linear_model.LinearRegression()
lm.fit(x_train,y_train)
lm.intercept_
lm.coef_
x.columns
coeff=pd.DataFrame(data=lm.coef_,index=x.columns,columns=['coefficients'])
print(coeff)
pred=lm.predict(x_test)
plt.scatter(y_test,pred)
sns.distplot(y_test-pred)
from sklearn.metrics import mean_squared_error
import math
rms=math.sqrt(mean_squared_error
                (pred,y_test)))