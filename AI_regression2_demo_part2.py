import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_boston()
print(data)

print(data.keys())

print(data.DESCR)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['MEDV'] = data.target
print(df.head())

plt.hist(df['MEDV'], 25)
plt.xlabel("MEDV")
plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(df['RM'], df['MEDV'])
plt.xlabel("RM")
plt.ylabel("MEDV")

plt.subplot(1, 2, 2)
plt.scatter(df['LSTAT'], df['MEDV'])
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()

X = pd.DataFrame(df[['LSTAT', 'RM']], columns=['LSTAT', 'RM'])
y = df['MEDV']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

print(X_train.shape)
print(X_test.shape)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
print("RMSE=", rmse, "R2=", r2)

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)
print("RMSE (test)=", rmse_test, "R2 (test)=", r2_test)

print(X_test, y_test_predict)
