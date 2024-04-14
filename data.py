import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import statsmodels.api as sm

path='Desktop/sample/50_Startups.csv'  # add your path to the csv file here
df = pd.read_csv(path,encoding='utf-8')
df.head()

df=df.rename(columns={'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms'},inplace=False)

df.head()

Investment=df.loc[:,['rd','ad','ms']]
Investment.head()

Investment['Total Investment']= Investment.sum(axis=1)

Investment

Investment['Total Investment'].mean()

df['Profit'].mean()

df.info()

df.shape

df.isna().sum()

df.corr()

sns.set(rc={'figure.figsize':(15,8)})

sns.heatmap(df.corr(),annot=True,cmap='Blues')

sns.scatterplot(x="rd",y="Profit",data=df)
plt.show()

sns.scatterplot(x="ad",y="Profit",data=df)
plt.show()

sns.scatterplot(x="ms",y="Profit",data=df)
plt.show()

df.hist()

sns.pairplot(data=df)
plt.show()

df.describe().T

outliers=['Profit']
plt.rcParams['figure.figsize']=[8,8]
sns.boxplot(data=df[outliers],orient="v",palette="Set2",width=0.7)

plt.title('Outlier Variable Distribution')
plt.ylabel('Profit Range')
plt.xlabel('Continuous Variable')

plt.show()

X=df.drop("Profit",axis=1)
Y=df["Profit"]

X.head()

Y.head()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42,shuffle=True)

X_train[0:10]

X_test[0:10]

Y_train[0:10]

Y_test[0:10]

lm=LinearRegression()

model=lm.fit(X_train,Y_train)
Y_pred=lm.predict(X_test)

df_comp=pd.DataFrame({'Actual Values':Y_test,'Estimates': Y_pred})
df_comp

MAE=mean_absolute_error(Y_test,Y_pred)
MAE

MSE=mean_squared_error(Y_test,Y_pred)
MSE

RMSE=math.sqrt(MSE)
RMSE

model.score(X,Y)

new_data=pd.DataFrame({'rd':165000,'ad':90000,'ms':350000}, index=[1])

Profit=model.predict(new_data)
print("Prediction in startup as per the model is:",Profit)

print("Range of profit lies between", (Profit-RMSE),"to", (Profit+RMSE))

