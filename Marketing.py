import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl

df = pd.read_csv('Marketing_Data.csv')
df.head()

print('----------------------------------------------------------------------')

plt.scatter(df.youtube, df.sales, color = 'blue')
plt.xlabel('youtube')
plt.ylabel("sales")
plt.show()

print('----------------------------------------------------------------------')

plt.scatter(df.facebook, df.sales, color = 'blue')
plt.xlabel('facebook')
plt.ylabel("sales")
plt.show()

print('----------------------------------------------------------------------')

plt.scatter(df.newspaper, df.sales, color = 'blue')
plt.xlabel('newspaper')
plt.ylabel("sales")
plt.show()


print('----------------------------------------------------------------------')


msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

from sklearn import linear_model
reg = linear_model.LinearRegression()
trainx = np.asanyarray(train[['youtube', 'facebook', 'newspaper']])
trainy = np.asanyarray(train[['sales']])
reg.fit(trainx, trainy)
print(reg.coef_)
print(reg.intercept_)


print('----------------------------------------------------------------------')


x = np.asanyarray(test[['youtube', 'facebook', 'newspaper']])
y = np.asanyarray(test[['sales']])
y_hat = reg.predict(test[['youtube', 'facebook', 'newspaper']])

print(reg.score(y,x))

print('----------------------------------------------------------------------')


# its how to predict your data
print(reg.predict([[33.22, 350.72, 50.96]]))
