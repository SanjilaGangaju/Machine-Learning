import numpy as np  
import pandas as pd 
import math
from sklearn.linear_model import LinearRegression
 
df=pd.read_csv("test_scores.csv")
x=df.math
y=df.cs

iteration=100
n=len(x)
lr=0.0002 #learning rate
def descent_parser(x,y):
    m=0
    b=0
    for i in range(iteration):
      y_predicted = m*x+b
      cost=1/n*sum([val**2 for val in y-y_predicted])
      
      md=-2/n*sum(x*(y-y_predicted))
      bd=2/n*sum(y-(y_predicted))

      m=m-lr*md
      b=b-lr*bd
      print("m{},b{},cost{},iteration{}".format(m,b,cost,i))
      cost_previous=0
      if math.isclose(cost,cost_previous, rel_tol=1e-20):
            cost_previous=cost
    return m,b

m, b = descent_parser(x,y)
print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

def predict():
    df=pd.read_csv('test_scores.csv')
    r=LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_[0],r.intercept_

m_sklearn, b_sklearn = predict()
print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))