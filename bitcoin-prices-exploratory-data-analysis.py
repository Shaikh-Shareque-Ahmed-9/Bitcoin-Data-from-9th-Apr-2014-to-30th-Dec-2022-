#!/usr/bin/env python
# coding: utf-8

#  **Importing Some Important Libraries**

# In[3]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 


# **Data processing:**

# In[4]:


df=pd.read_csv('BTC_Data_final.csv')


# In[5]:


df.info()


# In[6]:


df['Date']=pd.to_datetime(df['Date'])


# In[7]:


plt.style.use('seaborn-dark')


# In[8]:


plt.style.use('seaborn-dark-palette')


# **Feature Engineering**

# In[9]:


from pandas_profiling import ProfileReport


# In[10]:


profile = ProfileReport(df, title="Pandas Profiling Report")


# In[11]:


profile.to_widgets()


# **Exploratory data analysis:**

# In[12]:


df.plot(x='Date',y='priceUSD',kind='line',figsize=(12,8),lw=2.5,title='Bitcoin price over the years')


# In[13]:


btcm=pd.pivot_table(df,index=df['Date'].dt.year,values='mining_profitability',aggfunc='mean')
btcm=btcm.sort_values(by='Date', ascending=True)
btcm=btcm.head(10)
btcm.plot(kind='line',lw=3.5,figsize=(12,8),title='Mining profitability over the years')


# In[14]:


mnth=pd.pivot_table(df,index=[df['Date'].dt.year], values='size',aggfunc='mean')
mnth=mnth.sort_values(by='Date',ascending=True)
mnth.plot(kind='area',figsize=(12,8),title='Market size',)


# In[15]:


btcm1=pd.pivot_table(df,index=df['Date'].dt.year,values='transactionvalue',aggfunc='mean')
btcm1=btcm1.sort_values(by='Date', ascending=True)
btcm1=btcm1.head(10)
btcm1.plot(kind='line',lw=3.5,figsize=(12,8),title='Average transaction value over the years')


# In[16]:


df.plot(x='Date',y=['marketcap'],kind='area',lw=1.5,figsize=(14,8),title='Bitcoin Marketcap')


# In[17]:


df.plot(x='Date',y=['confirmationtime'],kind='scatter',lw=1.5,figsize=(12,8),title='Date/confirmation time correlation')


# In[18]:


df.plot(x='marketcap',y='priceUSD',kind='scatter',lw=1.5,figsize=(12,8),title='Marketcap / Price correlation')


# In[19]:


df.plot(x='sentbyaddress',y='activeaddresses',kind='scatter',lw=1.5,figsize=(12,8),title='Correlation between active adress and BTC sent')


# In[20]:


df.corr()


# In[21]:


prr=pd.pivot_table(df,index=df['Date'].dt.year,values='priceUSD',aggfunc='max')
prr=prr.sort_values(by='Date', ascending=True)
prr=prr.head(10)
prr


# In[22]:



prr.plot(kind='barh',figsize=(12,8),title='Highest BTC price',edgecolor='black',)


# In[23]:


prr2=pd.pivot_table(df,index=df['Date'].dt.year,values='tweets',aggfunc='mean')
prr2=prr2.sort_values(by='Date', ascending=True)
prr2=prr2.head(10)
prr2


# In[24]:


prr2.plot(kind='line',lw=2.5,figsize=(12,8),title='Average tweets over the years')


# In[25]:


prr3=pd.pivot_table(df,index=df['Date'].dt.year,values='google_trends',aggfunc='mean')
prr3=prr3.sort_values(by='Date', ascending=True)
prr3=prr3.head(16)
prr3


# In[26]:


prr3.plot(kind='line',lw=2.5,figsize=(12,8),title='Average Google trends over the years')


# In[27]:


prr4=pd.pivot_table(df,index=df['Date'].dt.year,values='confirmationtime',aggfunc='mean')
prr4=prr4.sort_values(by='Date', ascending=True)
prr4=prr4.head(10)
prr4.plot(kind='line',lw=3.5,figsize=(12,8),title='Average Confirmation time over the years')


# In[28]:


df.tail()

