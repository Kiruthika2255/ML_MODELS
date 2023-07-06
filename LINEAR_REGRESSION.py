#TO IMPLEMENT LINEAR REGRESSION

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

temp=[20,25,30,35,40]
ice=[13,21,25,35,38]

#.T method simply flips array SO temperature data in first column
x=np.array([temp]).T
y=np.array(ice)

rmodel=LinearRegression()
rmodel=rmodel.fit(x,y)

# Get the slope and intercept of the regression line
r_slope=rmodel.coef_
print('model slope',r_slope)

r_intercept=rmodel.intercept_
print('model intercept ',r_intercept)

pred=rmodel.predict(x)
print('prediction value',pred)

rmse=np.sqrt(mean_squared_error(y,pred))
print('model mean squared error value',rmse)

r2=rmodel.score(x,y)
print('model squared value',r2)


plt.scatter('temperature','ice sales',marker='*',edgecolors='r')
plt.plot(temp,pred)
plt.show()
