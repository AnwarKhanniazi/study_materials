# Ridge-and-Lasso-Regression


Assume that we have a model which is very accurate, therefore the error of our model will be low, meaning a low bias and low variance. Similarly we can say that if the variance increases, the spread of our data point increases which results in less accurate prediction. And as the bias increases the error between our predicted value and the observed values increases.

As we add more and more parameters to our model, its complexity increases, which results in increasing variance and decreasing bias, i.e., overfitting. So we need to find out one optimum point in our model where the decrease in bias is equal to increase in variance. In practice, there is no analytical way to find this point. So how to deal with high variance or high bias?
To overcome underfitting or high bias, we can basically add new parameters to our model so that the model complexity increases, and thus reducing high bias.
Now, how can we overcome Overfitting for a regression model?
Basically there are two methods to overcome overfitting,
        ●	Reduce the model complexity
        ●	Regularization


# REGULARIZATION:

In regularization, what we do is normally we keep the same number of features, but reduce the magnitude of the coefficients. For this purpose, we have different types of regression techniques which uses regularization to overcome this problem.
## 1. RIDGE REGRESSION:
    Ridge Regression is a regularization method that tries to avoid overfitting, penalizing large coefficients through the L2 Norm. For this reason, it is also called L2 Regularization.
    You can see that, as we increase the value of alpha, the magnitude of the coefficients decreases, where the values reaches to zero but not absolute zero.
    So, now you have an idea how to implement it but let us take a look at the mathematics side also. Till now our idea was to basically minimize the cost function, such that    values predicted are much closer to the desired result.
    Now take a look back again at the cost function for ridge regression.

                               min(summation(yi-yi^)^2)+lamda*[slope]^2
 
    While minimizing SSE using ridge regression this L2 penalty term makes coefficients to shrink
    Here if you notice, we come across an extra term, which is known as the penalty term. λ given here, is actually denoted by alpha parameter in the ridge function. So by changing the values of alpha, we are basically controlling the penalty term. Higher the values of alpha, bigger is the penalty and therefore the magnitude of coefficients are reduced.

**Important Points:**\
  *●	It shrinks the parameters, therefore it is mostly used to prevent multicollinearity*.\
  *●	It reduces the model complexity by coefficient shrinkage*.\
  *●	It uses L2 regularization technique.*


Now let us consider another type of regression technique which also makes use of regularization.


##2. Lasso regression:

            LASSO (Least Absolute Shrinkage Selector Operator), is quite similar to ridge, but let's understand the difference between them.
            After comparing we can see that, both the rmse and the value of R-square for our model will be increased. Therefore, lasso model is predicting better than both linear and ridge.(here i have chosen the case where Lasso regression R square value is high)
            We can see that as we increased the value of alpha, coefficients were approaching towards zero, but if you see in case of lasso, even at smaller alpha’s, our coefficients are reducing to absolute zeroes. Therefore, lasso selects the only some feature while reduces the coefficients of others to zero. This property is known as feature selection and which is absent in case of ridge.
            Mathematics behind lasso regression is quite similar to that of ridge only difference being instead of adding squares of theta, we will add absolute value of slope.
   
                            min(summation(yi-yi^)^2)+lamda*||slope||
                        
            Here too, λ is the hypermeter, whose value is equal to the alpha in the Lasso function.
            What this L1 penalty term does is, it not only shrinks coefficients but shrinks sum of them to zero and that is very useful for feature selection
 
**Important Points:**\
  *●	It uses L1 regularization technique.*\
  *●	It is generally used when we have more features because it automatically does feature selection.*
  
# Dataset:
      Housing 

# Programming
      Python
  
  
**The Codes regarding this Ridge and Lasso Regression model are *comparision between Linear, Ridge, Lasso Regresssions* with dataset of housing is present in this Repository in detail**

 
