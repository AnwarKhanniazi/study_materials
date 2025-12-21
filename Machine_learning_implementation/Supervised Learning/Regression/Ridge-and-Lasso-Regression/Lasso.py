# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:27:44 2020

@author: madis
"""


# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
housing = pd.read_csv("housing.csv")

# to get top 6 rows
housing.head(6) # to get top n rows use cars.head(10)

# Correlation matrix 
cor_mat = housing.corr()

# there isn't much high correlation among the independent variables
# Chance of having multicollinearity problem is less
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(housing)

# columns names
housing.columns
housing.shape
# Basic EDA
plt.scatter(x=np.arange(506),y=housing.CRIM)
plt.scatter(x=housing.CRIM,y=housing.MEDV,c="b");plt.xlabel("CRIM");plt.ylabel("MEDV")
plt.hist(housing.CRIM) # complete right skewed 
plt.hist(housing.MEDV) # close to normal distribution

# Statistical measurements
housing.describe()
#pd.tools.plotting.scatter_matrix(housing) #; -> also used for plotting all in one graph

# Checking whether we have any missing values or not 
housing.isnull().sum() # there are no missing values 

housing.head(4)

# Boxplots of all the columns
housing.boxplot()

# Histograms of all the columns 
housing.hist()

# preparing model considering all the variables using sklearn library
from sklearn.linear_model import LinearRegression
         
# Preparing model                  
LR1 = LinearRegression()
LR1.fit(housing.iloc[:,:13],housing.MEDV)
# Getting coefficients of variables               
LR1.coef_
LR1.intercept_

# Adjusted R-Squared value
LR1.score(housing.iloc[:,:13],housing.MEDV) # 0.74064
pred1 = LR1.predict(housing.iloc[:,:13])

# Rmse value
np.sqrt(np.mean((pred1-housing.MEDV)**2)) # 4.6791

# Residuals Vs Fitted Values
plt.scatter(x=pred1,y=(pred1-housing.MEDV));plt.xlabel("Fitted");plt.ylabel("Residuals");plt.hlines(y=0,xmin=0,xmax=60)
# Checking normal distribution 
plt.hist(pred1-housing.MEDV)

# Predicted Vs MEDV
plt.scatter(x=pred1,y=housing.MEDV);plt.xlabel("Predicted");plt.ylabel("Actual")
#plt.bar(height = pd.Series(LR1.coef_),left = housing.columns[:13])

# When we look at the weights assigned to each independent columns we see that there is high magnitude assigned for NOX
# But there is high correlation is existing between output and RM variable 
np.corrcoef(housing.NOX,housing.MEDV) # -0.427
np.corrcoef(housing.RM,housing.MEDV) # 0.695

### Let us split our entire data set into training and testing data sets
from sklearn.model_selection import train_test_split
train,test = train_test_split(housing,test_size=0.2)

### Preparing Ridge regression model for getting better weights on independent variables 
from sklearn.linear_model import Ridge

RM1 = Ridge(alpha = 0.4,normalize=True)
RM1.fit(train.iloc[:,:13],train.MEDV)
# Coefficient values for all the independent variables
RM1.coef_
RM1.intercept_
#plt.bar(height = pd.Series(RM1.coef_),left=pd.Series(housing.columns[:13]))
RM1.alpha # 0.05
pred_RM1 = RM1.predict(train.iloc[:,:13])
# Adjusted R-Squared value 
RM1.score(train.iloc[:,:13],train.MEDV) # 0.7342
# RMSE
np.sqrt(np.mean((pred_RM1-train.MEDV)**2)) # 4.8227


### Running a Ridge Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,100,0.05)
for i in alphas:
    RM = Ridge(alpha = i,normalize=True)
    RM.fit(train.iloc[:,:13],train.MEDV)
    R_sqrd.append(RM.score(train.iloc[:,:13],train.MEDV))
    train_rmse.append(np.sqrt(np.mean((RM.predict(train.iloc[:,:13]) - train.MEDV)**2)))
    test_rmse.append(np.sqrt(np.mean((RM.predict(test.iloc[:,:13]) - test.MEDV)**2)))
    
    
#### Plotting train_rmse,test_rmse,R_Squared values with respect to alpha values


# Alpha vs R_Squared values
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")

# Alpha vs train rmse
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("train_rmse")

# Alpha vs test rmse
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("test_rmse")
#plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))

# We got minimum R_Squared value at small alpha values 


# Let us prepare Lasso Regression on data set
from sklearn.linear_model import Lasso
LassoM1 = Lasso(alpha = 0.01,normalize=True)
LassoM1.fit(train.iloc[:,:13],train.MEDV)
# Coefficient values for all the independent variables
LassoM1.coef_
LassoM1.intercept_
#plt.bar(height = pd.Series(LassoM1.coef_),left=pd.Series(housing.columns[:13]))
LassoM1.alpha # 0.05
pred_LassoM1 = LassoM1.predict(train.iloc[:,:13])
# Adjusted R-Squared value 
LassoM1.score(train.iloc[:,:13],train.MEDV) # 0.12
# RMSE
np.sqrt(np.mean((pred_LassoM1-train.MEDV)**2)) # 4.951


### Running a LASSO Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,30,0.05)
for i in alphas:
    LRM = Lasso(alpha = i,normalize=True,max_iter=500)
    LRM.fit(train.iloc[:,:13],train.MEDV)
    R_sqrd.append(LRM.score(train.iloc[:,:13],train.MEDV))
    train_rmse.append(np.sqrt(np.mean((LRM.predict(train.iloc[:,:13]) - train.MEDV)**2)))
    test_rmse.append(np.sqrt(np.mean((LRM.predict(test.iloc[:,:13]) - test.MEDV)**2)))
    
    
#### Plotting train_rmse,test_rmse,R_Squared values with respect to alpha values

# Alpha vs R_Squared values
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")

# Alpha vs train rmse
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("train_rmse")

# Alpha vs test rmse
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("test_rmse")
#plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))

# We got minimum R_Squared value at small alpha values 
# from this we can say applying the simple linear regression technique is giving better results than Ridge and Lasso
# alpha tends 0 it indicates that Lasso and Ridge approximates to normal regression techniques 
