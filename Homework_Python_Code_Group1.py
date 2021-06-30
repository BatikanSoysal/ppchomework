#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


sample = pd.read_excel("Sample Submission.xlsx")


# In[3]:


store = pd.read_csv("store.csv")
store.head(100)


# In[4]:


dummies = pd.get_dummies(store[["PromoInterval"]])


# In[5]:


store_df = pd.concat([store, dummies[['PromoInterval_Jan,Apr,Jul,Oct', 'PromoInterval_Feb,May,Aug,Nov',
                                      'PromoInterval_Mar,Jun,Sept,Dec']]], axis = 1)


# In[6]:


store_df.head()


# In[7]:


store_df.drop(["PromoInterval"], axis = 1, inplace = True)


# In[8]:


store_df


# In[9]:


store_df["Competition"] = np.where(store_df["CompetitionOpenSinceMonth"].isna(), 0, 1)


# In[10]:


store_df.insert(3, "CompetitionON", store_df["Competition"])


# In[11]:


store_df.drop("Competition", axis = 1, inplace = True)


# In[12]:


store_df.rename(columns={"CompetitionON": "Competition"})


# In[13]:


store_df


# In[14]:


dummies = pd.get_dummies(store[["StoreType", "Assortment"]])
dummies


# In[15]:


store_df2 = pd.concat([store_df, dummies[['StoreType_a', 'StoreType_b',
                                      'StoreType_c', 'Assortment_a', 'Assortment_b']]], axis = 1)


# In[16]:


store_df = store_df2.copy()


# In[17]:


store_df = store_df.rename(columns={"CompetitionON": "Competition"})


# In[18]:


store_df.drop(["Assortment", "StoreType"], axis = 1, inplace = True)


# In[19]:


deneme = store_df[0:55]


# In[20]:


store_df = deneme.copy()


# In[21]:


train = pd.read_excel("Train Set.xlsx")
train = train.drop("Customers", axis = 1)
train.head()


# In[22]:


dummies = pd.get_dummies(train[["StateHoliday", "SchoolHoliday"]])


# In[23]:


train_df = pd.concat([train, dummies[['StateHoliday_a', 'StateHoliday_b',
                                      'StateHoliday_c']]], axis = 1)


# In[24]:


train_df.drop("StateHoliday", axis = 1, inplace = True)


# In[25]:


train_df


# In[26]:


df = pd.merge(train_df, store_df, how = 'inner', on = 'Store')


# In[27]:


#This is our store-train merged data.

df


# In[28]:


#Changing the date data to day/week/month data for better regression, since the month of a year or a specific week of a month
#may change sales value.

import datetime
df_day_list = []
for day in range(len(df)):
    df_day_list.append(int(df["Date"].loc[day].strftime("%d")))

df_month_list = []    
for month in range(len(df)):
    df_month_list.append(int(df["Date"].loc[month].strftime("%m")))

df_year_list = [] 
for year in range(len(df)):
    df_year_list.append(int(df["Date"].loc[year].strftime("%y")))


# In[29]:


df["Day"] = df_day_list
df["Month"] = df_month_list
df["Year"] = df_year_list


# In[30]:


df.drop("Date", axis = 1, inplace = True)


# In[31]:


df


# In[32]:


#Let's check for multicollinearity on our variables:

from statsmodels.stats.outliers_influence import variance_inflation_factor
dfdrop = df.drop(["Sales", "PromoInterval_Mar,Jun,Sept,Dec"], axis = 1)
vif_data = pd.DataFrame() 
vif_data["feature"] = dfdrop.columns                                                         
vif_data["VIF"] = [variance_inflation_factor(dfdrop.dropna().values, i) 
                          for i in range(len(dfdrop.columns))] 

print(vif_data)

#After checking for Multicollinearity, we have realised that PromoInterval_Mar,Jun,Sept,Dec was causing problems. 
#It fell into dummy variable trap so we dropped that column.

#We can also see that since none of our stores is StoreType B, and none of the Assortments are Assortment B,
#They are showing as NaN.


# In[33]:


df.drop("PromoInterval_Mar,Jun,Sept,Dec", axis = 1, inplace = True)


# In[34]:


df


# # At this point, we realized that we actually need to convert a lot more of the variables given to dummy variables, since most of them are not quantitative, instead categorical. We will check for multicollinearity one more time after these steps.

# In[35]:


df.columns


# In[36]:


df.isnull().sum()


# In[37]:


dummies = pd.get_dummies(df["Promo2SinceWeek"], prefix='Promo2SinceWeek', dummy_na = True)
dummies


# In[38]:


df_deneme = pd.concat([df, dummies], axis = 1)
df_deneme.drop("Promo2SinceWeek", axis = 1, inplace = True)
df_deneme


# In[39]:


dummies = pd.get_dummies(df["Promo2SinceYear"], prefix='Promo2SinceYear', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Promo2SinceYear", axis = 1, inplace = True)
df_deneme


# In[40]:


dummies = pd.get_dummies(df["CompetitionOpenSinceMonth"], prefix='CompetitionOpenSinceMonth', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("CompetitionOpenSinceMonth", axis = 1, inplace = True)
df_deneme


# In[41]:


dummies = pd.get_dummies(df["CompetitionOpenSinceYear"], prefix='CompetitionOpenSinceYear', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("CompetitionOpenSinceYear", axis = 1, inplace = True)
df_deneme


# In[42]:


dummies = pd.get_dummies(df["DayOfWeek"], prefix='DayOfWeek', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("DayOfWeek", axis = 1, inplace = True)
df_deneme


# In[43]:


df_deneme.columns


# In[44]:


dummies = pd.get_dummies(df["Day"], prefix='Day', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Day", axis = 1, inplace = True)

dummies = pd.get_dummies(df["Month"], prefix='Month', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Month", axis = 1, inplace = True)

dummies = pd.get_dummies(df["Year"], prefix='Year', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Year", axis = 1, inplace = True)

df_deneme


# In[45]:


df_deneme.columns


# In[46]:


dummies = pd.get_dummies(df["Store"], prefix='Store', dummy_na = True)

df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Store", axis = 1, inplace = True)

df_deneme


# # It looks like we got all the quantitative values to dummy variables, let's check for multicollinearity or NaN vif values and drop them.

# In[47]:


#It is recommended to not run this line since it will take a long time to pass and this is just to check for multicollinearity.
#Let's check for multicollinearity on our variables:
#This might take a while since we have 182 columns to work with.

vif_data = pd.DataFrame() 
vif_data["feature"] = df_deneme.drop("Sales", axis=1).columns                                                         
vif_data["VIF"] = [variance_inflation_factor(df_deneme.drop("Sales", axis=1).values, i) 
                          for i in range(len(df_deneme.drop("Sales", axis=1).columns))] 

print(vif_data)


# In[48]:


pd.options.display.max_rows = 1000
print(vif_data)

#We have got a lot of infs so let's try to drop some of the collineared ones.


# In[49]:


df_deneme2 = df_deneme.drop(['Store_55.0', 'Store_54.0', 'Store_53.0', 'Year_15.0', 'Month_12.0', 'Day_1.0',
                'DayOfWeek_nan', 'DayOfWeek_1.0', 'CompetitionOpenSinceYear_2015.0', 'CompetitionOpenSinceMonth_12.0',
                'Promo2SinceYear_2015.0'], axis = 1)


# In[50]:


# Last iteration did not do much so let's check our MSE and MAE values with this data set too.

# Before the multicollinearity correction iterations, we had values of:

"""
Regression: 

MSE: 1835970.3402729058
MAE: 989.7401177320178

ANN: 

MSE: 1408200.5
MAE: 783.6431884765625
"""


# In[51]:


# After dropping some columns and checking for Errrors, we have these values:

"""
Regression:
MSE: 1835941.948704593
MAE: 989.7220200516896

ANN:

MSE: 1471432.25
MAE: 804.4038696289062
"""

#So we can say not removing those features might be better since we got a higher MSE on ANN.


# In[52]:


y = df_deneme["Sales"]
X = df_deneme.drop(["Sales"], axis = 1)
X_backup = df_deneme.drop(["Sales"], axis = 1)
from sklearn.model_selection import train_test_split
# Splitting the merged dataset into train and validation datasets with %70 to %30
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=666)

print(y_train.shape, y_validation.shape)
print(X_train.shape, X_validation.shape)


# In[53]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# In[54]:


#Let's try the linear regression with validation to see what error value we are going to get.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
y_validation_pred = model.predict(X_validation)
print("MSE:", mean_squared_error(y_validation_pred, y_validation))

#Let's find our MAE value:
absolute_errors = []
for i in range(len(y_validation_pred)):
    absolute_errors.append(abs(y_validation_pred[i]-y_validation.iloc[i]))
absolute_errors.sort()
print("MAE:", sum(absolute_errors)/len(y_validation_pred))

#Let's plot a scatter plot to see the prediction accuracy
plt.scatter(y_validation_pred, y_validation)
plt.plot(np.arange(max(y_validation)), c = "r")
plt.legend(['Linear Line'], loc='best')
plt.show()



# ## Let's try to use ANN for better results. 

# In[55]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras
print(X_train.shape, X_validation.shape, y_train.shape, y_validation.shape)


# In[59]:


#Our ANN model with 32 input neurons, a hidden layer with 32 neurons and an output layer with 1:

model = Sequential([Dense(32, activation='relu'), Dense(32, activation='relu'), Dense(1, activation='relu')],)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist = model.fit(X_train, y_train,  batch_size=64, epochs=100, validation_data=(X_validation, y_validation))
model.evaluate(X_validation, y_validation)


# In[60]:


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[61]:


#Scatter plot for our fitted model's predicted sales values on our validation set vs our real sales values on validation set.

y_pred = model.predict(X_validation)
y_pred = pd.Series(y_pred[:,0], name = "Sales")
plt.scatter(y_pred, y_validation)
plt.plot(np.arange(max(y_validation)), c = "r")
absolute_errors = []
for i in range(len(y_pred)):
    absolute_errors.append(abs(y_pred.iloc[i]-y_validation.iloc[i]))
absolute_errors.sort()
print(sum(absolute_errors)/len(y_pred))
plt.legend(['Linear Line'], loc='best')
plt.show()


# In[62]:


#If we manually set the sales values of our predictions which have the Open column as 0 to 
# equal to 0, we can reduce our error a little more.

def manual_set_to_0():
    for i in range(len(y_pred)):
        if X_validation.iloc[i, 1] == 0:
            y_pred.iloc[i] = 0

#manuel_set_to_0()

#As this might be considered cheating, we are not going to do this.


# In[63]:


#Sorted values comparison
plt.plot(np.arange(len(y_validation)), y_validation.sort_values())
plt.plot(np.arange(len(y_pred)), y_pred.sort_values())


# In[64]:


#Plot to see our absolute error values
absolute_errors = []
for i in range(len(y_pred)):
    absolute_errors.append(abs(y_pred.iloc[i]-y_validation.iloc[i]))
absolute_errors.sort()
print(sum(absolute_errors)/len(y_pred))
plt.plot(np.arange(len(absolute_errors)), absolute_errors)


# ## Let's Try to use Ridge Regression

# In[65]:


from sklearn.linear_model import Ridge


# In[66]:


alphas = np.linspace(5,-2,100)
alpha_values = []
score = []
for i in alphas:
    ridge_model = Ridge(alpha = i)
    ridge_model.fit(X_train, y_train)      
    y_pred = ridge_model.predict(X_validation) 
    alpha_values.append(i)
    score.append(mean_squared_error(y_validation, y_pred))
alpha_values.sort()
score.sort()
rand_dict = {"Alpha values": alpha_values, "MSE": score}
ridge_scores = pd.DataFrame(rand_dict)


# In[67]:


ridge_scores


# ## As we can see, regularization is not helping that much. 
# ## And our ANN model is so much more accurate than our regression models.

# ## Now we are going to convert the requested test_set data to the same way we converted our test_set:

# In[68]:


test_set = pd.read_excel("Test Set.xlsx")
test_set.head()


# In[69]:


dummies = pd.get_dummies(test_set[["StateHoliday", "SchoolHoliday"]], dummy_na = True)
test_set = pd.concat([test_set, dummies[['StateHoliday_a', "StateHoliday_0", "StateHoliday_nan"]]], axis = 1)
test_set.drop("StateHoliday", axis = 1, inplace = True)

test_set.head()


# In[70]:


df_test = pd.merge(test_set, store_df, how = 'inner', on = 'Store')


# In[71]:


df_test.head()


# In[72]:


import datetime
df_day_list = []
for day in range(len(df_test)):
    df_day_list.append(int(df_test["Date"].loc[day].strftime("%d")))

df_month_list = []    
for month in range(len(df_test)):
    df_month_list.append(int(df_test["Date"].loc[month].strftime("%m")))

df_year_list = [] 
for year in range(len(df_test)):
    df_year_list.append(int(df_test["Date"].loc[year].strftime("%y")))

df_test["Day"] = df_day_list
df_test["Month"] = df_month_list
df_test["Year"] = df_year_list

df_test.drop("Date", axis = 1, inplace = True)


# In[73]:


df_test.head()


# In[74]:


df_test.columns


# In[75]:


#Putting extra dummies we did after realizing we might need them:

#Promo2SinceWeek
dummies = pd.get_dummies(df_test["Promo2SinceWeek"], prefix='Promo2SinceWeek', dummy_na = True)
df_deneme = pd.concat([df_test, dummies], axis = 1)
df_deneme.drop("Promo2SinceWeek", axis = 1, inplace = True)
print(df_deneme)

#Promo2SinceYear
dummies = pd.get_dummies(df_test["Promo2SinceYear"], prefix='Promo2SinceYear', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Promo2SinceYear", axis = 1, inplace = True)


#CompetitionOpenSinceMonth
dummies = pd.get_dummies(df_test["CompetitionOpenSinceMonth"], prefix='CompetitionOpenSinceMonth', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("CompetitionOpenSinceMonth", axis = 1, inplace = True)


#CompetitionOpenSinceYear
dummies = pd.get_dummies(df_test["CompetitionOpenSinceYear"], prefix='CompetitionOpenSinceYear', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("CompetitionOpenSinceYear", axis = 1, inplace = True)


#DayOfWeek
dummies = pd.get_dummies(df_test["DayOfWeek"], prefix='DayOfWeek', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("DayOfWeek", axis = 1, inplace = True)


#Date:
dummies = pd.get_dummies(df_test["Day"], prefix='Day', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Day", axis = 1, inplace = True)

dummies = pd.get_dummies(df_test["Month"], prefix='Month', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Month", axis = 1, inplace = True)

dummies = pd.get_dummies(df_test["Year"], prefix='Year', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Year", axis = 1, inplace = True)


#Store
dummies = pd.get_dummies(df_test["Store"], prefix='Store', dummy_na = True)
df_deneme = pd.concat([df_deneme, dummies], axis = 1)
df_deneme.drop("Store", axis = 1, inplace = True)


# In[76]:


print(df_deneme.columns)


# In[77]:


print(X_backup.columns)


# In[78]:


#To see which columns are we missing and add them with values of 0:
myList = list(set(X_backup.columns) - set(df_deneme.columns))
myList


# In[79]:


df_deneme.set_index("Id", inplace = True)


# In[80]:


#Dropping the columns not available on train set:
myList2 = set(df_deneme.columns) - set(X_backup.columns)

df_deneme.drop(myList2, axis = 1, inplace = True)


# In[81]:


#To create and add 0 values to missing columns:
for i in myList:
    df_deneme[i] = np.zeros(len(df_deneme['Open']))


# In[82]:


df_deneme


# In[83]:


df_deneme.sort_values(by = 'Id')


# In[84]:


#To check for missing columns
print(set(X_backup.columns) - set(df_deneme.columns))
print(set(df_deneme.columns) - set(X_backup.columns))


# In[85]:


#Our columns are matched. Now we can predict:
X_test = df_deneme.sort_values(by = 'Id')
X_test_backup = X_test
min_max_scaler = preprocessing.MinMaxScaler()
X_test = min_max_scaler.fit_transform(X_test)

y_test_predicted = model.predict(X_test)
#This is going to be our result, which we will use to submit.

y_test_predicted


# In[86]:


sample_sub = pd.read_excel("Sample Submission.xlsx")

print(sample_sub.count())
print(len(y_test_predicted))


# In[87]:


idx = X_test_backup.index
idx


# In[88]:


y_test_predicted_series = pd.Series(y_test_predicted[:,0], name = 'Sales')
y_test_predicted_series
y_test_predicted_dict = { 'Id' : idx, 'Sales' : y_test_predicted_series}
y_test_predicted_df = pd.DataFrame(y_test_predicted_dict)

#This is going to be our submission data.
y_test_predicted_df


# In[89]:


y_test_predicted_df['Sales'].mean()


# In[90]:


y.mean()


# In[91]:


#As seen above, our model probably did not make a very goodjob on predicting the test values since that mean is so much less
#than the mean of the test values. But still, this was the best we could do.


# In[92]:


#Exporting our data. You might skip this part too.

y_test_predicted_df.to_excel(r'Homework_Sales_Forecast_Group1.xlsx', index = False, header=False)


# In[ ]:




