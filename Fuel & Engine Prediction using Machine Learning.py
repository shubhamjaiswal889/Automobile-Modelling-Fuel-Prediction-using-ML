#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, skew

import warnings
warnings.filterwarnings("ignore")
sns.set()


# In[10]:


import os
for dirname, _, filenames in os.walk(r'C:\Users\shubham.kj\Downloads\auto-mpg.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[11]:


train = pd.read_csv(r'C:\Users\shubham.kj\Downloads\auto-mpg.csv')


# In[12]:


train.sample(5)


# In[13]:


print(train.info())


# In[14]:


train[train["horsepower"] == "?"]


# In[15]:




train.describe().T


# # Feature Engineering

# In[16]:


train["horsepower"] = train["horsepower"].replace("?", np.NaN).astype("float64")


# In[17]:


train_corr=train.corr().abs().unstack().sort_values(kind = "quiksort", ascending = False).reset_index()

train_corr.rename(columns = {"level_0": "Feature A",
                            "level_1": "Feature B",
                            0:"Correlation Coefs."}, inplace = True)

train_corr[train_corr["Feature A"] == "horsepower"].style.background_gradient(cmap = "coolwarm")


# In[18]:


train.groupby(['displacement'], sort = False)["horsepower"].apply(lambda x: x.fillna(x.mean()))
train['horsepower'] = train.groupby(['cylinders'], sort=False)['horsepower'].apply(lambda x: x.fillna(x.mean()))


# In[19]:


train.isna().sum()


# In[20]:


numerical_feat = train.select_dtypes(exclude = "object")
categorical_feat = train.select_dtypes(include = "object")

print("Numeric Features are   : ", *numerical_feat)
print("Categoric Features are : ", *categorical_feat)


# Check for skewness

# In[21]:


for column in numerical_feat.columns:
    plt.figure(figsize = (8,5))
    sns.distplot(train[column], fit = norm)
    plt.show()


#         Origin and Cylinders are far from normal distribution. But when I looked at the data set, these features were actually categorical values. That's why I'm ignoring these two. I will change them as categoric in a soon
#         Weight, Displacement, Horsepower features has positive skew, I need  to deal with it
#         I'm not gonna handle "mpg" because it is target feature
# 
# Now let's change the data types of Origin and Cylinders categorically then update the numerical_feat and categorical_feat variable with new ones

# In[22]:


train["origin"] = train["origin"].astype(str)
train["cylinders"] = train["cylinders"].astype(str)

numerical_feat = train.select_dtypes(exclude = "object")
categorical_feat = train.select_dtypes(include = "object")


# In[23]:


def checkSkewness(df):
  """
   - greater than 1 positive skewness
   - less than 1 negative skewness
  
  """
  skewed_feat = df.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
  skewness = pd.DataFrame(skewed_feat, columns = ["Skew Value"])

  return skewness.style.background_gradient(cmap='summer')


# In[24]:


checkSkewness(numerical_feat)


# As can be seen from the table, I will apply Log Transformation to get the "Horsepower", "Displacement", "weight" features from skewness.

# In[25]:


skew_feats = ["weight", "displacement","horsepower"]

train[skew_feats] = np.log1p(train[skew_feats])


# and check again
checkSkewness(train[skew_feats])


# They are better off now
# 

# In[26]:


categorical_feat.sample(5)


# # CARS PREDICTION

# In[27]:


train["car name"].value_counts()


# In[28]:


train = train.drop("car name", axis = 1)


# Cylinders

# In[29]:


train["cylinders"].value_counts(normalize = True)


# Half of our data set consists of 4-cylinder vehicles
# 

# In[30]:


plt.figure(figsize = (8,6))
sns.countplot(data = train, x = "cylinders")
plt.title("Cylinders")
plt.show()


# In[31]:


sns.displot(data = train, x = "mpg", hue = "origin",kind="kde");
plt.title("Kernel Density Estimation of MPG vs ORIGIN")
plt.xlabel("Miles Per Gallon")
plt.show()


# In[32]:


fig, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='origin', y="mpg", data=train)
plt.axhline(train.mpg.mean(),color='r',linestyle='dotted',linewidth=3)

plt.show()


# # Co-relation Matrix 

# In[33]:


train_corr = train.corr()

mask = np.triu(np.ones_like(train_corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(15, 6))

sns.heatmap(train_corr, annot=True,fmt='.2f',mask=mask, cmap="coolwarm", ax=ax);


# 
# There is a very high correlation between Weight and Displacement. This means that these two fefatures are almost the same.
# We can consider removing either of them from the model.
# 

# # Label Encoding

# In[34]:


train = pd.get_dummies(train) 


# In[35]:


train.head()


# # Data Modelling

# In[36]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

import lightgbm as lgb


# # Train & Test Splitting 

# In[37]:


x = train.drop('mpg', axis = 1)
y = train["mpg"]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .33, random_state = 42)


# # Scaling

# In[38]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Linear Regression

# In[39]:


lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)
y_predict_reg = lr_reg.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, y_predict_reg))
lr_rmse


# # Random Forest Generator

# In[40]:


rf_reg = RandomForestRegressor().fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rf_rmse


# # Lasso Regressor

# In[41]:


lasso_reg = Lasso().fit(X_train, y_train)

y_pred = lasso_reg.predict(X_test)

lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_rmse


# # Elastic Net Regressor

# In[42]:


eNet = ElasticNet().fit(X_train, y_train)

y_pred = eNet.predict(X_test)

eNet_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
eNet_rmse


# # LGBM Regressor

# In[43]:


lgb_reg = lgb.LGBMRegressor().fit(X_train, y_train)

y_pred = lgb_reg.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lgb_rmse


# In[44]:


model_names = ["LASSO","LGBM","RANDOM FR","LINEAR","ELASTICNET"]

models = {"Model":model_names,
          "RMSE":[lasso_rmse, lgb_rmse, rf_rmse,lr_rmse,eNet_rmse]}

model_performance = pd.DataFrame(models)

model_performance.sort_values(by = "RMSE",kind='quicksort', ascending=True).style.background_gradient(cmap='summer')


# ElasticNet shows the biggest improvement after hyperparameter optimization. LGBM gives the worst performance after tuning. Sure, larger captive param_grid results could have been much different, but it's a very time-consuming process.
# 
# As a result, Random Forest Regressor was the best algorithm in Tunned models, and LGBM was the best model in basic models.
# 

# In[ ]:




