#!/usr/bin/env python
# coding: utf-8

# In[1]:


# linear algebra
import numpy as np 
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 


# In[2]:


# reading dataset
df=pd.read_csv("BTC_Data_final.csv")


# In[3]:


df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.set_index("Date", inplace = True)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


# Visulising the price of BTC 30 day average basis
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
sns.set_style('whitegrid')
df['priceUSD'].plot(figsize=(12,6),label='price')
df['priceUSD'].rolling(window=30).mean().plot(label='30 Day Avg')# Plotting the 
plt.legend()
plt.show()


# In[8]:


df.isnull().sum()


# In[9]:


# ploting the no of tractions on 30 average basis
sns.set()
sns.set_style('whitegrid')
df['transactions'].plot(figsize=(12,6),label='transactions')
df['transactions'].rolling(window=30).mean().plot(label='30 Day Avg')# Plotting the 
plt.legend()
plt.show()


# In[10]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot = True, cmap= 'YlGnBu', fmt= '.2f')
plt.show()


# In[11]:


# Visulising the price of BTC 30 day average basis
features=list(df.columns)
for i in features:
    sns.set()
    sns.set_style('whitegrid')
    df[i].rolling(window=30).mean().plot(figsize=(12,6),label=i)
    plt.legend()
    plt.show()


# In[12]:


df = df.assign(Change=pd.Series(df.priceUSD.div(df.priceUSD.shift())))
df['Change'].plot(figsize=(20,8))
plt.show()


# In[13]:


df = df.assign(expanding_mean=pd.Series(df['priceUSD'].expanding(1).mean()))
df['expanding_mean'].plot(figsize=(20,8))
plt.show()


# In[14]:


df['lag_1'] = df['priceUSD'].shift(1)
df['lag_2'] = df['priceUSD'].shift(2)
df['lag_3'] = df['priceUSD'].shift(3)
df['lag_4'] = df['priceUSD'].shift(4)
df['lag_5'] = df['priceUSD'].shift(5)
df['lag_6'] = df['priceUSD'].shift(6)
df['lag_7'] = df['priceUSD'].shift(7)


# In[15]:


df = df.assign(Return=pd.Series(df.Change.sub(1).mul(100)))
df['Return'].plot(figsize=(20,8))
plt.show()


# In[16]:


df.priceUSD.pct_change().mul(100).plot(figsize=(20,6))
plt.show()


# In[17]:


df = df.assign(Mean=pd.Series(df['priceUSD'].rolling(window=30).mean()))
df['Mean'].plot(figsize=(20,8),label='mean price')
df['priceUSD'].plot(label='original')
plt.legend()
plt.show()


# In[18]:


# summary stats
print(df["priceUSD"].describe())


# In[19]:


# histogram plot
df["priceUSD"].hist()
plt.show()


# In[20]:


# prepare situation
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from matplotlib import pyplot
def moving_average_(data):
    X = data
    window = 3
    history = [X[i] for i in range(window)]
    test = [X[i] for i in range(window, len(X))]
    predictions = list()
    # walk forward over time steps in test
    for t in range(len(test)):
        length = len(history)
        yhat = np.mean([history[i] for i in range(length-window,length)])
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # zoom plot
    pyplot.plot(test[0:100],label="Original")
    pyplot.plot(predictions[0:100], color='red',label="Prediction")
    plt.legend()
    pyplot.show()


# In[21]:


moving_average_(df["priceUSD"].values)


# In[22]:


df_train = df[df.index < "2019"]
df_valid = df[df.index >= "2019"]


# In[23]:


df.columns


# In[24]:


important_feature_=['size', 'sentbyaddress', 'transactions',
       'mining_profitability', 'sentinusd', 'transactionfees',
       'median_transaction_fee', 'confirmationtime', 'marketcap',
       'transactionvalue', 'mediantransactionvalue', 'tweets', 'google_trends',
       'fee_to_reward', 'activeaddresses', 'top100cap', 'Change',
       'expanding_mean', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6',
       'lag_7', 'Return', 'Mean']


# In[25]:


get_ipython().system(' pip install pmdarima')


# In[26]:


from pmdarima import auto_arima
model = auto_arima(df_train.priceUSD,
                   trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.priceUSD)


# In[27]:


forecast = model.predict(n_periods=len(df_valid))
forecast = pd.DataFrame(forecast).reset_index()
forecast.rename(columns= {0: 'forecasted','index':'date'},inplace=True)


# In[28]:


import matplotlib.pyplot as plt
plt.plot(forecast['date'], forecast['forecasted'],label='forecasted')
plt.plot(df_valid.index, df_valid['priceUSD'],label='real')
plt.legend()
plt.show()


# In[ ]:




