#!/usr/bin/env python
# coding: utf-8

# # Time Series Analysis-Electric Production

# #### Import Required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


# #### Import Data

# In[2]:


data=pd.read_csv(r"C:\Users\MDHONDIB\Desktop\Python Self TRainingTraining\Electric_Production.csv")
data.head()


# #### Convert Date variable to Date

# In[3]:


data['DATE'] = pd.to_datetime(data['DATE'])
print(data.head())


# #### Rename Variable Names

# In[4]:


data.rename(columns={'DATE': 'date', 'IPG2211A2N': 'value'}, inplace=True)


# #### Set Date Variable as index

# In[5]:


data.index = data['date']
del data['date']
print(data.head())


# #### Convert table to Dataframe

# In[6]:


df = pd.DataFrame(data)
df.head()


# #### Plot Time Series Graph

# In[7]:


'''def plot_df(df, x, y, title="", xlabel='Date', ylabel='value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.value, title='Monthly Electric Production')
'''


# In[8]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df.index, y=df.value,
                    mode='lines+markers',
                    name='Training',
                    line=dict(color="#0000ff")))
fig.update_layout(
    title="Electric Procution-Time Series Plot",
    xaxis_title="Date",
    yaxis_title="Value",
    #legend_title="Legend Title"
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ))


# #### Check Stationarity using ADF  Test
# Ho:Series is Non Stationary vs H1: Series is stationary

# In[9]:


# ADF Test
result = adfuller(df.value.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# #### Split Data into Traning & Testing

# In[10]:


train = df[df.index < pd.to_datetime("2012-12-31", format='%Y-%m-%d')]
test = df[df.index > pd.to_datetime("2012-12-31", format='%Y-%m-%d')]
train.head()
test.tail()


# #### Plot Train Test Data

# In[11]:


'''plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('Value')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")
plt.show()
'''


# In[12]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=train.index, y=train.value,
                    mode='lines+markers',
                    name='Training',
                    line=dict(color="#0000ff")))
fig.add_trace(go.Scatter(x=test.index, y=test.value,
                    mode='lines+markers',
                    name='Testing',
                    line=dict(color="#dbae0b")))
fig.update_layout(
    title="Electric Procution Plot",
    xaxis_title="Date",
    yaxis_title="Value",
    #legend_title="Legend Title"
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ))


# #### Fit Arima Model

# In[13]:


'''ARIMAmodel = ARIMA(train["value"], order = (2, 1, 2)) # Specify ARIMA Parameters 
ARIMAmodel = ARIMAmodel.fit()
y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)  # 95% Confidence Interval
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.head()
y_pred_df.index = test.index

#Show actual vs forecasting
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('Value')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")
plt.plot(y_pred_df["Predictions"], color='Yellow', label = 'ARIMA Predictions')
plt.legend()
import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["value"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)
'''


# #### Decomposition

# In[14]:


result = seasonal_decompose(df)
fig = plt.figure()  
fig = result.plot()


# #### Check Stationarity for original Series

# In[15]:


result=adfuller(df.value.dropna())
print(f'ADF Statistics:{result[0]}')
print(f'p-value:{result[1]}')


# #### Check Stationarity for Diff Series

# In[16]:


result=adfuller(df.value.diff().dropna())
print(f'ADF Statistics:{result[0]}')
print(f'p-value:{result[1]}')


# #### Plot ACF and PACF 

# In[17]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[18]:


fig, (ax1, ax2)=plt.subplots(2,1,figsize=(8,8))

plot_acf(df,lags=14, zero=False, ax=ax1)
plot_pacf(df,lags=14, zero=False, ax=ax2)
plt.show()


# #### Fit Autoarima Model

# In[22]:


from pmdarima.arima import auto_arima
import pmdarima


# In[23]:


model=pmdarima.auto_arima(df,seasonal=True, trace=True, error_action='ignore',m=12, stepwise=True)


# #### Choose model from Autoarima

# In[24]:


model=SARIMAX(train,order=(1,1,2),  seasonal_order=(1, 0, 1, 12))
model=model.fit()


# In[25]:


model.summary()


# #### Check Dignostics

# In[26]:


# Create the 4 diagostics plots
model.plot_diagnostics(figsize=(8,8))
plt.show()


# ### Predict future value using test Data

# In[27]:


start=len(train)
end=len(train)+len(test)-1
prediction = model.predict(start=start,end=end)
prediction.head()
prediction=pd.DataFrame(prediction) # for plotly library we convert into data frame normal graph not required
prediction.head()


# In[28]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=train.index, y=train.value,
                    mode='lines+markers',
                    name='Training Data',
                    line=dict(color="#0000ff")))
fig.add_trace(go.Scatter(x=test.index, y=test.value,
                    mode='lines+markers',
                    name='Testing Data',
                    line=dict(color="#ebb028")))
fig.add_trace(go.Scatter(x=prediction.index, y=prediction.predicted_mean,
                    mode='lines+markers',
                    name='Test_Prediction',
                    line_color='#0e7536'))
fig.update_layout(
    title="Electric Procution Prediction Training/Testing",
    xaxis_title="Date",
    yaxis_title="Value",
    #legend_title="Legend Title"
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ))


# #### RSME for Test and predicted

# In[35]:


from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["value"].values, prediction["predicted_mean"]))
print("RMSE: ",arma_rmse)


# ### Confidence Limit 

# In[30]:


confi_int_p=model.conf_int()
lower_limits_p=confi_int_p.iloc[:,0]
upper_limits_p=confi_int_p.iloc[:,1]


# ### Forecast future using historical data

# In[31]:


# Make ARIMA forecast of next 10 values
forecast = model.predict(start=len(df),end=len(df)+30)
#forecast=pd.head()
forecast=pd.DataFrame(forecast) # for plotly library we convert into data frame normal graph not required
forecast.head()


# In[32]:


'''plt.figure(figsize=(14,5))
plt.plot(train,label='Actual values', color="blue", marker="o")
plt.plot(test, label='Test', color="green", marker="o")
plt.ylabel('Value')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")
#prediction.plot(legend=True,color="red", marker="o")
forecast.plot(legend=True,color="orange", marker="o")
#plt.fill_between(prediction[len(test)].index, lower_limits_p, upper_limits_p, alpha=0.1, color="green")
plt.axvline(prediction.index[-1], color="red", linestyle="--")
plt.legend()
'''


# In[33]:


fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=train.index, y=train.value,
                    mode='lines+markers',
                    name='Training Data',
                    line=dict(color="#0000ff")))
fig.add_trace(go.Scatter(x=test.index, y=test.value,
                    mode='lines+markers',
                    name='Testing Data',
                    line=dict(color="#ebb028")))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.predicted_mean,
                    mode='lines+markers',
                    name='Forecasting',
                    line=dict(color="#08a642")))
fig.update_layout(
    title="Electric Procution Prediction",
    xaxis_title="Date",
    yaxis_title="Value",
    #legend_title="Legend Title"
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ))


# In[ ]:




