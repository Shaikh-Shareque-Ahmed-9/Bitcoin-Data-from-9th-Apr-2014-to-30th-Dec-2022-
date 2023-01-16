#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing liberaries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
import seaborn as sns
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tabulate import tabulate
from pandas import read_csv
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# reading dataset
# This data set is collected using web scraping methods from https://bitinfocharts.com till 18th may 2021
df=pd.read_csv("BTC_Data_final.csv")


# In[3]:


df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.set_index("Date", inplace = True)


# In[4]:


df.head()


# In[5]:


df.tail(6)


# In[6]:


df.shape


# In[7]:


from pandas_profiling import ProfileReport


# In[8]:


profile = ProfileReport(df, title="Pandas Profiling Report")


# In[9]:


profile.to_widgets()


# In[10]:


# checking data set contains null values
df.isnull().values.any()


# In[11]:


missed = pd.DataFrame()
missed['column'] = df.columns

missed['percent'] = [round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns]

missed = missed.sort_values('percent',ascending=False)
print(missed)


# In[12]:


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
print(mem_usage(df))


# In[13]:


# Visulising the price of BTC 30 day average basis
sns.set()
sns.set_style('whitegrid')
df['priceUSD'].plot(figsize=(12,6),label='price')
df['priceUSD'].rolling(window=30).mean().plot(label='30 Day Avg')# Plotting the 
plt.legend()
plt.show()


# In[14]:


# ploting the no of tractions on 30 average basis
sns.set()
sns.set_style('whitegrid')
df['transactions'].plot(figsize=(12,6),label='transactions')
df['transactions'].rolling(window=30).mean().plot(label='30 Day Avg')# Plotting the 
plt.legend()
plt.show()


# In[15]:


# statistcs 
from tabulate import tabulate
info = [[col, df[col].count(), df[col].max(), df[col].min(),df[col].mean()] for col in df.columns]
#print(tabulate(info, headers = ['Feature', 'Count', 'Max', 'Min','Mean'], tablefmt = 'orgtbl'))


# In[16]:


df1=df.reset_index(drop=True)
X=df1.drop('priceUSD', 1)
X


# In[17]:


y=df1[["priceUSD"]]
y


# # Dropping those features which is highly correlated each other.

# In[18]:


# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.75
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]


# In[19]:


# Drop features 
X.drop(X[to_drop], axis=1,inplace=True)


# In[20]:


X_columns=list(X.columns)
y_columns=["priceUSD"]


# In[21]:


correlation_result={}
for i in range(len(X_columns)):
    correlation = X[X_columns[i]].corr(y["priceUSD"])
    correlation_result[X_columns[i]]=correlation
correlation_result=sorted(correlation_result.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)


# In[22]:


temp=[]
for i in correlation_result:
    temp.append(i[0])
X_train=X[temp]
X_train


# In[23]:


plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), annot = True, cmap= 'YlGnBu', fmt= '.2f')
plt.show()


# In[24]:


# Visulising the price of BTC 30 day average basis
features=list(X.columns)
for i in features:
    sns.set()
    sns.set_style('whitegrid')
    X[i].rolling(window=30).mean().plot(figsize=(12,6),label=i)
    plt.legend()
    plt.show()


# In[25]:


estimators=[]
estimators.append(['minmax',MinMaxScaler(feature_range=(-1,1))])
scale=Pipeline(estimators)
X_min_max=scale.fit_transform(X)
y_min_max=scale.fit_transform(y)


# In[26]:


from sklearn.decomposition import PCA
pca = PCA(random_state=0).fit(X_min_max)

plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.xlabel('number of components')
plt.ylabel('cumulative variance %')
plt.show()


# In[27]:


np.cumsum(pca.explained_variance_ratio_)


# In[28]:


train_data_ = df[list(X_train.columns)]
train_data_['priceUSD'] = df['priceUSD']


# In[29]:


train_data_


# In[30]:


train_data_.sort_index()['2010':'2022']["priceUSD"].plot(subplots=True, figsize=(15,10))
plt.savefig('bitcoin.png')
plt.show()


# # Percent change

# In[31]:


#train_data_['Change'] = train_data_.priceUSD.div(train_data_.priceUSD.shift())
train_data_ = train_data_.assign(Change=pd.Series(train_data_.priceUSD.div(train_data_.priceUSD.shift())))
train_data_['Change'].plot(figsize=(20,8))
plt.show()


# # Expanding mean

# In[32]:


#train_data_['expanding_mean'] = train_data_['priceUSD'].expanding(1).mean()
train_data_ = train_data_.assign(expanding_mean=pd.Series(train_data_['priceUSD'].expanding(1).mean()))
train_data_['expanding_mean'].plot(figsize=(20,8))
plt.show()


# # Lag feature

# In[33]:


train_data_['lag_1'] = train_data_['priceUSD'].shift(1)
train_data_['lag_2'] = train_data_['priceUSD'].shift(2)
train_data_['lag_3'] = train_data_['priceUSD'].shift(3)
train_data_['lag_4'] = train_data_['priceUSD'].shift(4)
train_data_['lag_5'] = train_data_['priceUSD'].shift(5)
train_data_['lag_6'] = train_data_['priceUSD'].shift(6)
train_data_['lag_7'] = train_data_['priceUSD'].shift(7)


# # Return

# In[34]:


train_data_ = train_data_.assign(Return=pd.Series(train_data_.Change.sub(1).mul(100)))
train_data_['Return'].plot(figsize=(20,8))
plt.show()


# In[35]:


train_data_.priceUSD.pct_change().mul(100).plot(figsize=(20,6))
plt.show()


# # Window functions

# In[36]:


train_data_ = train_data_.assign(Mean=pd.Series(train_data_['priceUSD'].rolling(window=30).mean()))
train_data_['Mean'].plot(figsize=(20,8),label='mean price')
train_data_['priceUSD'].plot(label='original')
plt.legend()
plt.show()


# # Time series decomposition and Random walks

# In[37]:


train_data_["priceUSD"].plot(figsize=(25,10))
plt.show()


# In[38]:


# Now, for decomposition...
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 25, 15
decomposed_train_data_ = sm.tsa.seasonal_decompose(train_data_["priceUSD"],period=365) # The frequncy is annual
figure = decomposed_train_data_.plot()
plt.show()


# In[39]:


train_data_.isnull().values.any()


# In[40]:


train_data_.dropna(axis = 0, how ='any',inplace=True)
train_data_.isnull().values.any()


# In[41]:


train_data_["priceUSD"].describe()


# In[42]:



from scipy import signal
detrended = signal.detrend(train_data_["priceUSD"].values)
plt.plot(detrended)
plt.title('detrended by subtracting the least squares fit', fontsize=16)


# In[43]:


# Plotting white noise
from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
series = Series(train_data_["priceUSD"])
# summary stats
print(train_data_["priceUSD"].describe())


# In[44]:


# histogram plot
series.hist()
plt.show()


# In[45]:


# autocorrelation
autocorrelation_plot(series)
plt.show()


# In[46]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(train_data_["priceUSD"],lags=20)
plt.show()


# In[47]:


# Plotting autocorrelation of white noise
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train_data_["priceUSD"],lags=150,alpha=0.05)
plt.show()


# The partial autocorrelation function shows a high correlation with the first lag and lesser correlation with the second and third lag. The autocorrelation function shows a slow decay, which means that the future values have a very high correlation with its past values.

# As we can see, the time series contains significant auto-correlations up through lags 130

# In[48]:


import statsmodels.stats.diagnostic as diag
diag.acorr_ljungbox(train_data_["priceUSD"], lags=[140], boxpierce=True)


# The value 256068.08656797 is the test statistic of the Box-Pierce test and 0.0 is its p-value as per the Chi-square(k=140) tables.
# As we can see, both p-values are less than 0.01 and so we can say with 99% confidence that the time series is not pure white noise.

# ## Random Walk

# In[49]:


# Augmented Dickey-Fuller test on volume of google and microsoft stocks 
#https://www.statsmodels.org/dev/_modules/statsmodels/tsa/stattools.html
from statsmodels.tsa.stattools import adfuller
adf = adfuller(train_data_["priceUSD"])
print("p-value : {}".format(float(adf[1])))


# ## Generating a random walk

# In[50]:


diff_Y_i = train_data_["priceUSD"].diff()
train_data_ = train_data_.assign(difference=pd.Series(diff_Y_i))

#drop the NAN in the first row
diff_Y_i = diff_Y_i.dropna()
diff_Y_i.plot()
plt.show()


# In[51]:


import statsmodels.stats.diagnostic as diag
diag.acorr_ljungbox(diff_Y_i, lags=[140], boxpierce=True)


# ## Stationarity

# In[52]:


# non stationary
decomposed_train_data_.trend.plot()
plt.show()


# In[53]:


# The new stationary plot
decomposed_train_data_.trend.diff().plot()
plt.show()


# In[54]:


train_data_


# # Modelling using statstools

# # autoregressive (AR) mode

# # Forecasting a simulated model

# In[55]:


print(train_data_.isnull().values.sum())
train_data_.dropna(axis = 0, how ='any',inplace=True)
print(train_data_.isnull().values.sum())


# In[56]:


# prepare situation
def moving_average_(data):
    X = data
    window = 3
    history = [X[i] for i in range(window)]
    test = [X[i] for i in range(window, len(X))]
    predictions = list()
    # walk forward over time steps in test
    for t in range(len(test)):
        length = len(history)
        yhat = mean([history[i] for i in range(length-window,length)])
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


# In[57]:


moving_average_(train_data_["priceUSD"].values)


# In[58]:


df_train = train_data_[train_data_.index < "2019"]
df_valid = train_data_[train_data_.index >= "2019"]


# In[59]:


def exponential_moving_():
    weights = np.arange(1,31) #this creates an array with integers 1 to 31 included
    weights
    wma10 = train_data_["priceUSD"].rolling(30).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    train_data_['30day_WMA'] = np.round(wma10, decimals=3)
    #sma10 = train_data_['priceUSD'].rolling(30).mean()
    temp = train_data_.dropna(how='any',axis=0) 
    print(sqrt(mean_squared_error(temp.priceUSD, temp['30day_WMA'])))
    plt.figure(figsize = (12,6))
    plt.plot(train_data_['priceUSD'], label="Price")
    plt.plot(wma10, label="30-Day WMA")
    #plt.plot(sma10, label="10-Day SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# In[60]:


exponential_moving_()


# # Exponential Moving Average

# In[61]:


def exponential_moving_average():
    ema30 = train_data_['priceUSD'].ewm(span=30).mean()
    train_data_['30_day_EMA'] = np.round(ema30, decimals=3)
    print(sqrt(mean_squared_error(train_data_.priceUSD, train_data_['30_day_EMA'])))
    plt.figure(figsize = (12,6))
    plt.plot(train_data_['priceUSD'], label="Price")
    plt.plot(ema30, label="30-Day WMA")
    #plt.plot(sma10, label="10-Day SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# In[62]:


exponential_moving_average()


# In[63]:



from pandas import read_csv
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
# prepare situation
X = train_data_["priceUSD"].values
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)):
	length = len(history)
	yhat = mean([history[i] for i in range(length-window,length)])
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


# # Prediction using ARIMA model

# In[64]:


#train_data_.columns
train_data_.fillna(method='ffill', inplace=True)
train_data_.fillna(method='backfill', inplace=True)


# In[65]:


train_data_.isnull().values.any()


# In[66]:


#train_data_.columns


# In[67]:


df_train = train_data_[train_data_.index < "2019"]
df_valid = train_data_[train_data_.index >= "2019"]


# In[68]:


pip install pmdarima --user


# In[69]:


get_ipython().system('pip install pmdarima')


# In[70]:


pip install --upgrade pip


# In[71]:


from pmdarima import auto_arima
model = auto_arima(df_train.priceUSD, exogenous=df_train,                   trace=False, error_action="ignore", suppress_warnings=True)
model.fit(df_train.priceUSD, exogenous=df_train)


# In[72]:


forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid)
forecast = pd.DataFrame(forecast).reset_index()
forecast.rename(columns= {0: 'forecasted','index':'date'},inplace=True)


# In[73]:


import matplotlib.pyplot as plt
plt.plot(forecast['date'], forecast['forecasted'],label='forecasted')
plt.plot(df_valid.index, df_valid['priceUSD'],label='real')
plt.legend()
plt.show()


# In[74]:


print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.priceUSD, forecast['forecasted'])))
print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.priceUSD, forecast['forecasted']))


# In[ ]:




