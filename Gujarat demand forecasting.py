# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:11:44 2020

@author: gauravrai
"""

'''Projet: Gujarat Load forecasting Model'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

def decompse_data(ts):
    decomposition=seasonal_decompose(ts,freq=7)
    trend=decomposition.trend
    seasonal=decomposition.seasonal
    residual=decomposition.resid
    ax=plt.figure(figsize=(15,10))
    plt.subplot(411)
    plt.plot(ts,label="Original",color='blue')
    plt.legend(loc='right',fontsize=8)
    plt.xticks(rotation=45,fontsize=8)
    plt.subplot(412)
    plt.plot(trend,label="Trend",color='Green')
    plt.legend(loc='right',fontsize=8)
    plt.xticks(rotation=45,fontsize=8)
    plt.subplot(413)
    plt.plot(seasonal,label="Seasonality",color='Red')
    plt.legend(loc='right',fontsize=8)
    plt.xticks(rotation=45,fontsize=8)
    plt.subplot(414)
    plt.plot(residual,label="Residual",color='Orange')
    plt.legend(loc='right',fontsize=8)
    plt.xticks(rotation=45,fontsize=8)
    ax.suptitle("Decomposition of Data: Trend Seasonality and Residual",fontsize=14)
    plt.tight_layout()
    plt.show()
    
def modelverification(ts,forecast1,actual1):
    
    forecast1=pd.DataFrame(forecast1)
    actual1=actual1.reset_index()
    forecast1.reset_index(inplace=True)
   
    forecast=forecast1[forecast1.columns[1]]
    actual=actual1[actual1.columns[1]]

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
   
    print("mape:",mape)
    print("me:",me)
    print("mae:",mae)
    print("mpe:",mpe)
    print("RMSE:",rmse)
    print("Correlation:",corr)
    
    #Validation through residuals
    resd_ar=pd.DataFrame(ts.resid)
    plt.figure(figsize=(10,5))
    plt.plot(resd_ar,color='red')
    plt.title("Residual plot of Model")
    plt.show()
    print("Mean of Residual is:\n")
    print(resd_ar.mean())
    
    plt.figure(figsize=(10,5))
    resd_ar.hist()
    plt.title("Histogram of Residual plot of Model")
    plt.show()

    '''-------------Lung-box test-----------'''
    ltest=sm.stats.acorr_ljungbox(ts.resid, lags=[10])
    print(ltest)
 
def visualization(ts):
    
    ts1=ts['03-04-2017':'30-06-2017']
    
    roll1=ts1['ACTUAL ENERGY (MWh)'].rolling(96).sum()
    plt.plot(roll1)
    plt.title("Daily Consumption pattern gujarat")
    
   
    ts1a= ts1.groupby(ts1['BLOCKNO'],as_index=False).aggregate({'ACTUAL ENERGY (MWh)':['mean']})
       
    ts1a['ACTUAL ENERGY (MWh)'].plot()
    plt.title('Block wise mean consumption (MWH)')
    plt.show()
    return(ts1a)    
 
def simpleexponeential(train,test):
     #Smoothing_level is alpha value   
    fitdata= SimpleExpSmoothing(train).fit(smoothing_level=0.5,optimized=False)
    fcast=fitdata.forecast(len(test))
    plt.figure(figsize=(18,8))
    plt.plot(train,label='train data',color='black')
    plt.plot(test,label='test data',color='green')
    plt.plot(fcast,label='forecast',color='red')
    plt.legend(loc='best')
    plt.title('Load Forecast using Simple Exponential Method',fontsize=15)
    plt.xlabel('day----->')
    plt.ylabel('Consumption in Mwh')
    plt.show()
    
    print("Verification of Simple Exponential Forecasting Model")
    modelverification(fitdata,fcast,test)
    return(fitdata)
    
def HLM_model(train,test):
    #alpha=smoothing_level and beta=smoothing slope
    fit1 = Holt(train).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
    fcast=fit1.forecast(len(test))
    plt.figure(figsize=(18,8))
    plt.plot(train,label='train data',color='black')
    plt.plot(test,label='test data',color='green')
    plt.plot(fcast,label='forecast',color='red')
    plt.legend(loc='best')
    plt.title('Load Forecast using HLM trend Method',fontsize=15)
    plt.xlabel('day----->')
    plt.ylabel('Consumption in Mwh')
    plt.show()
    print("Verification of HLM(trend) Forecasting Model")
    modelverification(fit1,fcast,test)
    return(fit1)

def HLM_winter_model(train,test):
    #alpha=smoothing_level and beta=smoothing slope

    fit1 = ExponentialSmoothing(train,seasonal_periods=150,trend='additive',seasonal='additive',damped=False).fit(optimized=True, use_boxcox=False, remove_bias=True)
    fcast=fit1.forecast(len(test))
    plt.figure(figsize=(18,8))
    plt.plot(train,label='train data',color='black')
    plt.plot(test,label='test data',color='green')
    plt.plot(fcast,label='forecast',color='red')
    plt.legend(loc='best')
    plt.title('Load Forecast using HLM winter Method',fontsize=15)
    plt.xlabel('day----->')
    plt.ylabel('Consumption in Mwh')
    plt.show()
    results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
    params = ['smoothing_level', 'smoothing_slope', 'damping_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
    results["Additive"]= [fit1.params[p] for p in params] + [fit1.sse]
    print(results)
    print("Verification of HLM winter Forecasting Model")
    modelverification(fit1,fcast,test)
    return(fit1)
    
    
'''------------------------ARIMA Model---------------------------------------------'''
def test_stationarity(ts):
    #Rolling Statistics
    rolmean=ts.rolling(1).mean()
    rolstd=ts.rolling(1).std()
    
    #Plotting Rolling Statistics
    fig,ax1=plt.subplots()
    ax1.plot(ts,color='blue',label='original')
    ax1.plot(rolmean,color='red',label='Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Performing Dicky-Fuller-Test (DFtest statistic more closer to -12.51 and P-value close to 0.01 means white noise and it is good to go)
    '''----H0: TimeSeries is non-stationary-----'''
    ts1=pd.DataFrame(ts)
    
    ts1=ts1.reset_index()
   
    ts2=ts1[ts1.columns[1]]
    
    print ('Results of Dickey-Fuller Test:')
    
    dftest=adfuller(ts2)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    print ('Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test:')
    df2test=kpss(ts2,regression='ct',nlags='auto')
    dfoutput1 = pd.Series(df2test[0:4], index=['KPSS Statistic','p-value','#Lags Used','critical'])
    print(dfoutput1)
    
    
def make_stationarity_diff(ts):
    ts_diff=ts.diff(1)
    ts_diff.dropna(inplace=True)
    test_stationarity(ts_diff)
    return(ts_diff)

def acf_pacf(ts):
    ax1=plt.figure(figsize=(15,10))
    lag_acf=acf(ts,nlags=20)
    lag_pacf=pacf(ts,nlags=20,method='ols')
    print(lag_acf)
    print(lag_pacf)
    plt.figure()
    plt.subplot(211)
    plot_acf(ts, ax=plt.gca())
    plt.subplot(212)
    plot_pacf(ts, ax=plt.gca())
    plt.show()
    '''
    #plot ACF
    plt.subplot(121)
    plt.plot(lag_acf,'b-')
    plt.axhline(y=0,linestyle='--',color='grey')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.title('Auto Correlation Function')
    
    # plot PACF
    plt.subplot(122)
    plt.plot(lag_pacf,'g-')
    plt.axhline(y=0,linestyle='--',color='grey')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.title('Partial Auto Correlation Function')
    plt.tight_layout()
    plt.show()'''
    
def ARIMA_model(train,test,p,i,q):
     model=ARIMA(train,order=(p,i,q),freq='D')
     res=model.fit()
     print(res.summary())
     plt.plot(train,label='training data')
     plt.plot(res.fittedvalues,color='red',linestyle='--',label='fitted value')
     plt.legend(loc='best')
     plt.title('ARIMA_Model')    
     plt.show()
     fc,se,conf=res.forecast(len(test),alpha=0.05)
     print(fc)
     # Make as pandas series
     fc_series = pd.Series(fc, index=test.index)
     lower_series = pd.Series(conf[:, 0], index=test.index)
     upper_series = pd.Series(conf[:, 1], index=test.index)
    
     #plot
     plt.figure(figsize=(15,8), dpi=100)
     plt.plot(train,label='training')
     plt.plot(test, label='actual',color='red')
     plt.plot(fc_series, label='forecast',color='black')
     plt.fill_between(lower_series.index, lower_series, upper_series,color='k', alpha=.15)
     plt.title('Forecast vs Actuals using ARIMA Model',fontsize=15)
     plt.legend(loc='upper left', fontsize=8)
     plt.show()
     
     modelverification(res,fc_series,test)
 
def srimax_model(train,test,p,i,q,sp,si,sq):
     model=sm.tsa.statespace.SARIMAX(train,order=(p,i,q),seasonal_order=(sp,si,sq,145),enforce_stationarity=True,enforce_invertibility=False)
     res=model.fit(method='bfgs')
     print(res.summary())
     
     plt.plot(train,label='training data')
     plt.plot(res.fittedvalues,color='red',linestyle='--',label='fitted value')
     plt.legend(loc='best')
     plt.title('SARIMAX_Model')    
     plt.show()
     fc=res.predict(start='2018-02-04',end='2018-12-31',dynamic=False)
     
     #plot
     plt.figure(figsize=(15,8), dpi=100)
     plt.plot(train,label='training')
     plt.plot(test, label='actual',color='red')
     plt.plot(fc, label='forecast',color='black')
     plt.title('Forecast vs Actuals using SARIMAX Model',fontsize=15)
     plt.legend(loc='upper left', fontsize=8)
     plt.show()
     
def forecast_model(model,train,test):
    fcast=model.forecast(200)
    plt.figure(figsize=(18,8))
    plt.plot(train,label='train data',color='black')
    plt.plot(test,label='test data',color='green')
    plt.plot(fcast,label='forecast',color='red')
    plt.legend(loc='best')
    plt.title('Load Forecast using HLM winter Method',fontsize=15)
    plt.xlabel('day----->')
    plt.ylabel('Consumption in Mwh')
    plt.show()
   
'''--------------------------------Main Program-------------------------'''

os.chdir(r'C:\Users\gauravrai\Desktop\IPBA\Time Series Model\Gujarat load forecasting')
data=pd.read_excel('Gujarat Power Demand.xlsx')

data.set_index(['DAY'],inplace=True)
data['2017-06']


k=data['ACTUAL ENERGY (MWh)'].resample('d').sum()
k.plot()
plt.xticks(rotation=45,fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel("Energy Consumption in Mwh------->")
plt.xlabel("Months--------------->")
plt.title("Daily Consumption pattern of Gujarat",fontsize=14)
plt.show()

train=pd.DataFrame(k[:'2018-01-28'])
test=pd.DataFrame(k['2018-02-04':])

a=1
while(a==1):
    decompse_data(train)
    print("Press 1 for Simple Exponential Model")
    print("Press 2 for HL Trend Model")
    print("Press 3 for HL Winter Model")
    print("Press 4 for Arima Model")
    print("Press 5 for Srimax Model")
    print("Press 6 to Exit")
    b=int(input("Enter your option:"))
    if (b==1):
       
       model=simpleexponeential(train,test)
       forecast_model(model,train,test)
       
    elif(b==2):
        
        model=HLM_model(train,test)
        forecast_model(model,train,test)
    elif(b==3):
        model=HLM_winter_model(train,test)
        forecast_model(model,train,test)
    elif(b==4):
        test_stationarity(train)
        ts_diff=make_stationarity_diff(train)
        acf_pacf(ts_diff)
        p=int(input("Enter P Value: "))
        i=int(input("Enter I Value: "))
        q=int(input("Enter q Value: "))
        model=ARIMA_model(train,test,p,i,q)
    elif(b==5):
        test_stationarity(train)
        ts_diff=make_stationarity_diff(train)
        acf_pacf(train)
        p=int(input("Enter P Value: "))
        i=int(input("Enter I Value: "))
        q=int(input("Enter q Value: "))
        sp=int(input("Enter seasonal P Value: "))
        si=int(input("Enter seasonal I Value: "))
        sq=int(input("Enter seasonal q Value: "))
        srimax_model(train,test,p,i,q,sp,si,sq)
        
    elif(b==6):
         print("Thanks for using the Model")
         a=2
