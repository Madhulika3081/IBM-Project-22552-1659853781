#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import itertools 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# import dataset
df = pd.read_csv('C:\\Users\\Madhulika\\Downloads\\Churn_Modelling.csv')
df.head()


# # Univariate Analysis

# In[8]:


df.shape


# In[9]:


df.groupby(['Geography']).count()


# In[12]:


freq_table=df.groupby(['Geography']).size().reset_index(name='Count').rename(columns={'Geography':'Geography'})
freq_table


# In[13]:


plt.bar(freq_table['Geography'],freq_table['Count'])
plt.show()


# In[14]:


freq_table['Count%']=freq_table['Count']/sum(freq_table['Count'])*100
freq_table


# # Categorical Variable Analysis

# In[18]:


df_plot=df.groupby(['IsActiveMember','Geography']).size().reset_index.pivot(columns='IsActiveMember',index='Geography',values=0)
df.plot.plot(x=df.plot_index,kind='bar',stacked=True)


# # Missing Values

# In[20]:


df.isnull().sum()


# In[28]:


df.shape


# In[29]:


df.dropna(how='any').shape


# ## Remove Outliers

# In[30]:


df.describe()


# # (BoxPlot)

# In[36]:


def plot_boxplot(df,ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()


# In[65]:


plot_boxplot(df,"Age")


# In[66]:


plot_boxplot(df,"CreditScore")


# In[67]:


def outliers(df,ft):
    Q1=df[ft].quantile(0.25)
    Q3=df[ft].quantile(0.25)
    IQR=Q3-Q1
    
    lower_bound=Q1-1.5*IQR
    upper_bound=Q1+1.5*IQR
    
    ls=df.index[(df[ft]<lower_bound) | (df[ft]>upper_bound)]
    return ls


# In[73]:


index_list=[]
for feature in['CreditScore','Age']:
    index_list.extend(outliers(df,feature))


# In[76]:


index_list


# In[77]:


def remove(df,ls):
    ls=sorted(set(ls))
    df=df.drop(ls)
    return df


# In[78]:


df_cleaned=remove(df,index_list)


# In[79]:


df_cleaned.shape


# In[80]:


plot_boxplot(df_cleaned,'CreditScore')


# In[81]:


plot_boxplot(df_cleaned,'Age')


# ## Splitting data into dependent and independent variables

# In[90]:


X=df.iloc[:,2:12]
X


# In[89]:


Y=df.iloc[:,12]
Y


# ## Splitting data into training and testing

# In[101]:


plt.scatter(df['Tenure'],df['EstimatedSalary'])


# In[99]:


plt.scatter(df['Age'],df['EstimatedSalary'])


# In[102]:


X = df[['Tenure','Age']]


# In[103]:


y = df['EstimatedSalary']


# In[104]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3) 


# In[105]:


X_train


# In[106]:


X_test


# In[107]:


y_train


# In[108]:


y_test


# # (linear regression model)
# 

# In[109]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)


# In[110]:


X_test


# In[111]:


clf.predict(X_test)


# In[112]:


y_test


# In[113]:


clf.score(X_test, y_test)


# # random_state argument
# 

# In[114]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
X_test


# In[ ]:




