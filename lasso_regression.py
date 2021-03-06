# -*- coding: utf-8 -*-
"""
# Lasso Regression Problem
"""

# imports the neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression,f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

"""### Importing the Data
 Pandas is an open-source, library providing high-performance,easy-to-use data manipulation and data analysis tools.
"""

# Data Fetch
#file=("C:/Users/Pawan Kataria/Downloads/housing.csv")
#df= pd.read_csv(file)
#df.head()
import pandas as pd
import io
from google.colab import files

uploaded = files.upload()
import io

df = pd.read_csv(io.BytesIO(uploaded['housing.csv']))

df.head()

"""### Feature Selection
 It is the process of reducing the number of input variables when developing a predictive model.Used to reduce the number of input variables to reduce the computational cost of modelling and,in some cases,to improve the performance of the model.
"""

# Selected Columns where X is independent variables and Y is dependent variable.
features=['CRIM',	'ZN',	'INDUS',	'CHAS',	'NOX',	'RM',	'AGE',	'DIS',	'RAD',	'TAX','PTRATIO','B','LSTAT']
target='MEDV'
# X & Y
X=df[features]
Y=df[target]

"""### Data Preprocessing
 Since the majority of the machine learning models in the Sklearn library doesn't handle string category data and Null value,we have to explicitly remove or replace null values.The below snippet have functions, which removes the null value if any exists.
"""

# Data Cleaning
def NullClearner(value):
	if(isinstance(value, pd.Series) and (value.dtype in ['float64','int64'])):
		value.fillna(value.mean(),inplace=True)
		return value
	elif(isinstance(value, pd.Series)):
		value.fillna(value.mode()[0],inplace=True)
		return value
	else:return value
x=X.columns.to_list()
for i in x:
	X[i]=NullClearner(X[i])
Y=NullClearner(Y)

"""### Correlation Matrix
 In order to check the correlation between the features, we will plot a correlation matrix. It is effective in summarizing a large amount of data where the goal is to see patterns.
"""

f,ax = plt.subplots(figsize=(18, 18))
matrix = np.triu(X.corr())
se.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)
plt.show()

"""### Multi-colinearity Test
 Dropping Highly Correlated Features to due similar features distributions

"""

def dropHighCorrelationFeatures(X):
        cor_matrix = X.corr()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if to_drop!=[]: return X.drop(to_drop, axis=1)
        else: return X
X=dropHighCorrelationFeatures(X)
X.head()

"""### Best Feature Selection
 selecting 'n' best feature on the basis of ANOVA or Univariate Linear Regression Test. where ANOVA is used for Classification problem and Univariate Linear Regression for Regression problems

"""

def get_feature_importance(X,Y,score_func):
    fit = SelectKBest(score_func=score_func, k=X.shape[1]).fit(X,Y)
    dfscores,dfcolumns = pd.DataFrame(fit.scores_),pd.DataFrame(X.columns)
    df = pd.concat([dfcolumns,dfscores],axis=1)
    df.columns = ['features','Score'] 
    df['Score']=MinMaxScaler().fit_transform(np.array(df['Score']).reshape(-1,1))
    result=dict(df.values)
    val=dict(sorted(result.items(), key=lambda item: item[1],reverse=False))
    keylist=[]
    for key, value in val.items():
        if value < 0.01: keylist.append(key)
    X=X.drop(keylist,axis=1)
    plt.figure(figsize = (12, 6))
    plt.barh(range(len(val)), list(val.values()), align='center')
    plt.yticks(range(len(val)),list(val.keys()))
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    return X
X=get_feature_importance(X,Y,score_func=f_regression)

"""### Data Rescaling
 Feature scaling or Data scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization
"""

columns=X.columns
X=StandardScaler().fit_transform(X)
X=pd.DataFrame(data = X,columns = columns)
X.head()

"""### Train & Test
 The train-test split is a procedure for evaluating the performance of an algorithm.The procedure involves taking a dataset and dividing it into two subsets.The first subset is utilized to fit/train the model.The second subset is used for prediction.The main motive is to estimate the performance of the model on new data.
"""

# Data split for training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

"""### Model

Linear Model trained with L1 prior as regularizer (aka the Lasso)

The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer non-zero coefficients, effectively reducing the number of features upon which the given solution is dependent. For this reason Lasso and its variants are fundamental to the field of compressed sensing.

#### Model Tuning Parameter
1. **alpha** -> Constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised.

2. **selection** -> If set to ???random???, a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to ???random???) often leads to significantly faster convergence especially when tol is higher than 1e-4.

3. **tol** -> The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.

4. **max_iter** -> The maximum number of iterations.
"""

# Model Initialization
model=Lasso()
model.fit(X_train,Y_train)

"""### Accuracy Metrics
 Performance metrics are a part of every machine learning pipeline. They tell you if you're making progress, and put a number on it. All machine learning models,whether it's linear regression, or a SOTA technique like BERT, need a metric to judge performance.
 1. R-squared is a statistical measure that represents the goodness of fit of a regression model. The ideal value for r-square is 1. The closer the value of r-square to 1, the better is the model fitted.
 2. Mean Absolute Error is a model evaluation metric used with regression models. The mean absolute error of a model with respect to a test set is the mean of the absolute values of the individual prediction errors on over all instances in the test set. Each prediction error is the difference between the true value and the predicted value for the instance.
 3. Mean squared error (MSE) measures the amount of error in statistical models. It assesses the average squared difference between the observed and predicted values. When a model has no error, the MSE equals zero. As model error increases, its value increases. The mean squared error is also known as the mean squared deviation (MSD).
"""

# Metrics
y_pred=model.predict(X_test)
print('R2 Score: {:.2f}'.format(r2_score(Y_test,y_pred)))
print('Mean Absolute Error {:.2f}'.format(mean_absolute_error(Y_test,y_pred)))
print('Mean Squared Error {:.2f}'.format(mean_squared_error(Y_test,y_pred)))
