import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import datetime

import os
import sys

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import matplotlib as mpl

import statsmodels.formula.api as smf
import matplotlib as mpl

from scipy import stats
from statistics import stdev
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

    ############################################################################################

def add_to_year(df, amount = 1):
    original_year = df['Date'].dt.year[0]

    replaced_year = original_year + amount
    new_df = df.copy()
    new_df['Date'] = new_df['Date'].mask(new_df['Date'].dt.year == original_year,\
                     new_df['Date'] + pd.offsets.DateOffset(year = replaced_year))
    return new_df


def panda_read_file(csv_file_name): 
    raw_df = pd.read_csv(csv_file_name)
    column_names = list(raw_df.columns)
    if 'day' in column_names: # this part was added because of the format of my csv file
        raw_df['Date'] = pd.to_datetime(raw_df['day'], format='%j').dt.strftime('%Y-%m-%d')
        raw_df['Date'] = pd.DatetimeIndex(raw_df['Date'])
        column_names.remove('day')
    if 'timestr' in column_names: # this part was added because of the format of my csv file
        raw_df['Time'] = pd.to_datetime(raw_df['timestr']).dt.strftime('%H:%M:%S')
        column_names.remove('timestr')
    return [raw_df, column_names]
    
    ############################################################################################
    
def Understand_data():
    csv_file_name = 'stockdata3 - Copy.csv'
    [semi_proc_df, stock_names] = panda_read_file(csv_file_name)
    semi_proc_df = add_to_year(semi_proc_df, 69)
    print(semi_proc_df.head())
    print(semi_proc_df.tail())
    print(semi_proc_df.describe())
    print(stock_names)
    return [semi_proc_df, stock_names]
    
[semi_proc_df, stock_names] = Understand_data()

    ############################################################################################
    
def replace_outliers_with_zero(semi_proc_stock_prices, non_outliers):
    stock_prices = np.copy(semi_proc_stock_prices)
    for i in range(stock_prices.shape[0]):
        for j in range(stock_prices.shape[1]):
            if not non_outliers[i, j]:
                stock_prices[i, j] = 0
    return stock_prices    
    
    
def find_outliers(semi_proc_stock_prices, non_outliers):
    stock_prices = np.copy(semi_proc_stock_prices)
    z_score_outliers(stock_prices, non_outliers, 3)
    iqr(stock_prices, non_outliers, 1.5)
    zero_outliers(stock_prices, non_outliers)
    #count_outliers(non_outliers)
    #replace_outliers_with_zero(stock_prices, non_outliers)
    #return stock_prices
        
    # records which ones have z-scores with absolute value more than coeff
def z_score_outliers(stock_prices, non_outliers, coeff = 3):
    for j in range(stock_prices.shape[1]):
        z_scores = stats.zscore(stock_prices[ : , j])
        non_outliers[ : , j] = np.logical_and(np.absolute(z_scores) <= coeff,\
                                                non_outliers[ : , j])
    
        
     # https://kanoki.org/2020/04/23/how-to-remove-outliers-in-python/
def iqr(stock_prices, non_outliers, coeff):
    for j in range(stock_prices.shape[1]):
        Q1 = np.quantile(stock_prices[ : , j], 0.25)
        Q3 = np.quantile(stock_prices[ : , j], 0.75)
        IQR = Q3-Q1
        lower_bound = Q1 - coeff*IQR
        upper_bound = Q3 + coeff*IQR
        between_intervals = np.logical_and(stock_prices[ :, j] <= upper_bound,\
                                            stock_prices[ :, j] >= lower_bound)
        non_outliers[ : , j] = np.logical_and(between_intervals, non_outliers[ : , j])        
        #df['ewm_alpha_1']=df['data'].ewm(alpha=0.1).mean()
        
def zero_outliers(stock_prices, non_outliers):
    non_outliers = np.logical_and(non_outliers, stock_prices != 0)   
    
    ############################################################################################    
    
def replace_zeros_with_averages(stock_prices, non_outliers):
    for j in range(stock_prices.shape[1]):
        prev = 0
        after = 0
        know_after = False
        for i in range(stock_prices.shape[0]):
            if non_outliers[i, j] and stock_prices[i, j] != 0: # for removing zeros
                prev = stock_prices[i, j]
                know_after = False
            else:
                if know_after:
                    stock_prices[i, j] = (prev+after)/2
                    continue
                k = i+1
                while k < non_outliers.shape[0] and not non_outliers[k, j]:
                    k += 1
                after = stock_prices[k, j]
                know_after = True
                stock_prices[i, j] = (prev+after)/2

    # plot and histogram of columns of np array stock_prices
def plotting_stuff(stock_prices, stock_names, text_to_add, plotting = True, histogram = True, first = 0, last = None):
    #titles = ["Stock a", "Stock b", "Stock c", "Stock d", "Stock e", "Stock f"]
    for j in range(stock_prices.shape[1]):
        current_column = stock_prices[first :last , j]
        panda_column = pd.DataFrame(current_column)
        print('column', j, panda_column.describe())
        if plotting:
            plt.plot(current_column)
            plt.xlabel("time")
            plt.ylabel("stock price")
            curr_title = "Stock " + stock_names[j] + " Line Graph "+ text_to_add
            plt.title(curr_title)
            plt.show()
        
        if histogram: 
            bins = 100  
            # plotting a histogram 
            plt.hist(current_column, bins, color = 'green',\
                    histtype = 'bar', rwidth = 0.8) 
            plt.xlabel('price of stock') 
            plt.ylabel('repetition')
            plt.yscale("log")
            curr_title = "Stock " + stock_names[j] + ' Histogram '+ text_to_add
            plt.title(curr_title)   
            plt.show() 

    ############################################################################################

def pre_processing(semi_proc_df, stock_names):
    #stock_columns = ['a', 'b', 'c', 'd', 'e', 'f']
        
    raw_stock_prices = semi_proc_df[stock_names].to_numpy()
    
    non_outliers = np.full((raw_stock_prices.shape[0], raw_stock_prices.shape[1]), True)
    semi_proc_stock_prices =  np.nan_to_num(raw_stock_prices)
    plotting_stuff(semi_proc_stock_prices, stock_names, "Before Processing")
    
    find_outliers(semi_proc_stock_prices, non_outliers)
    stock_prices = replace_outliers_with_zero(semi_proc_stock_prices, non_outliers)
    
    replace_zeros_with_averages(stock_prices, non_outliers)
    plotting_stuff(stock_prices, stock_names, "After Processing")
    
    for j in range(non_outliers.shape[1]):
        print(sum(non_outliers[:, j])/non_outliers.shape[0])
        

    df = semi_proc_df.copy()
    for j in range(stock_prices.shape[1]):
        col = stock_names[j]
        df[col] = stock_prices[ : , j]     
    return df

df = pre_processing(semi_proc_df, stock_names)

    ############################################################################################
    
# if interval_length is None, do it daily. Otherwise, interval_length is number of minutes.
def find_relevant_info(df, stock_names, interval_length = None):    
    df_list = []   
    #new_columns = ['Date', 'Time', 'Low', 'High', 'Open', 'Close']
    stock_prices = df[stock_names].to_numpy()

    for j in range(stock_prices.shape[1]):
        
        new_df = {'Date':[], 'Time':[], "Low":[], "High":[], "Open":[], "Close":[]}
        i = 1
        prev = 0
        while i < len(df['Date']):
            if df['Date'][i] != df['Date'][i-1]: # i is start of a new day
                if not interval_length: # doing stuff daily
                    new_interval = stock_prices[prev:i, j] # stock prices on a day
                    new_df['Date'].append(df['Date'][prev])
                    new_df['Time'].append(df['Time'][prev])
                    append_extra_information(new_df, new_interval)
                    
                else:
                    for k in range(prev, i, interval_length):
                        # stock prices on a day with length interval length. 
                        # the last ind should be at most i so interval lies within a day
                        new_interval = stock_prices[k:min(k+interval_length, i), j]
                        new_df['Date'].append(df['Date'][prev])
                        new_df['Time'].append(df['Time'][k])
                        append_extra_information(new_df, new_interval)
                prev = i

            i += 1
            
        if not interval_length: # doing stuff daily
            new_interval = stock_prices[prev:i, j]
            new_df['Date'].append(df['Date'][prev])
            new_df['Time'].append(df['Time'][prev])
            append_extra_information(new_df, new_interval)
        else:
            for k in range(prev, i, interval_length):
                new_interval = stock_prices[k:min(k+interval_length, i), j]
                new_df['Date'].append(df['Date'][prev])
                new_df['Time'].append(df['Time'][k])
                append_extra_information(new_df, new_interval)
        
        df_list.append(pd.DataFrame.from_dict(new_df))            
    return df_list

def append_extra_information(dictionary, interval):
    dictionary['Low'].append(interval.min())
    dictionary['High'].append(interval.max()) 
    dictionary['Open'].append(interval[0])
    dictionary['Close'].append(interval[-1])

   #############################################################################################
   
def find_rates(df_list, based_on_column_name = 'Close', logarithmic = True):
    for j in range(len(df_list)):
        daily_log_return_rate = [None]
        nightly_return_rate = [None]
        log_return_rate = [0] 
        for i in range(1, len(df_list[j][based_on_column_name])):
            v_final   = df_list[j][based_on_column_name][i]
            v_initial = df_list[j][based_on_column_name][i-1]
            ratio = v_final/v_initial    
            if logarithmic:
                new_number = np.log(ratio)*100
            else:
                new_number = (ratio-1)*100
            log_return_rate.append(new_number)
            if df_list[j]['Date'][i] == df_list[j]['Date'][i-1]:
                daily_log_return_rate.append(new_number)
                nightly_return_rate.append(None)
            else:
                daily_log_return_rate.append(None)
                nightly_return_rate.append(new_number)
            
        df_list[j]['Log Return Rate'] = log_return_rate
        df_list[j]['Daily Log Return Rate'] = daily_log_return_rate
        df_list[j]['Nightly Log Return Rate'] = nightly_return_rate
        
    #############################################################################################
    
def find_volatility(df_list, period_length_in_minutes = None, period_length_in_days = 1, interval_length = 1):
    vol_df_list = []
    minutes_in_day = 391 
    days_in_year = 252
    coeff = np.sqrt(minutes_in_day*days_in_year/interval_length)
    if period_length_in_minutes:
        if period_length_in_minutes < interval_length:
            print("can't have period length in minutes less than interval length.")
            return
        block_length = period_length_in_minutes//interval_length
        for j in range(len(df_list)):
            new_dict = {'Date':[], 'Time':[], 'Volatility':[]}
            i = 1
            prev = 1
            while i < len(df_list[j]['Date']):
                if df_list[j]['Date'][i] != df_list[j]['Date'][i-1]: #end of a day
                    for k in range(prev, i, block_length):
                        new_standard_dev = np.std(df_list[j]['Daily Log Return Rate'][k:min(k+block_length, i)])
                        new_dict['Date'].append(df_list[j]['Date'][min(k+block_length, i)-1])
                        new_dict['Time'].append(df_list[j]['Time'][min(k+block_length, i)-1])
                        new_dict['Volatility'].append(new_standard_dev*coeff)
                    prev = i+1
                i += 1
            for k in range(prev, i, block_length):
                new_standard_dev = np.std(df_list[j]['Daily Log Return Rate'][k:min(k+block_length, i)])
                new_dict['Date'].append(df_list[j]['Date'][min(k+block_length, i)-1])
                new_dict['Time'].append(df_list[j]['Time'][min(k+block_length, i)-1])
                new_dict['Volatility'].append(new_standard_dev*coeff)
            my_columns = ['Date', 'Time', 'Volatility']
            vol_df_list.append(pd.DataFrame(new_dict, columns = my_columns))
        
    else:
        for j in range(len(df_list)):
            new_dict = {'Date':[], 'Volatility':[]}
            (day_numbers, day_start_ind) = np.unique(df_list[j]['Date'], return_index = True)
            prev = 0
            for k in range(0, len(day_numbers), period_length_in_days):
                if k == 0:
                    continue
                i = day_start_ind[k]
                new_standard_dev = np.std(df_list[j]['Log Return Rate'][prev:i])
                new_dict['Date'].append(df_list[j]['Date'][i])
                new_dict['Volatility'].append(new_standard_dev*coeff)
                prev = i
            my_columns = ['Date', 'Volatility']
            vol_df_list.append(pd.DataFrame(new_dict, columns = my_columns))
            
    return vol_df_list   

   #############################################################################################
   
    # this function is written following the example in
    # http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016
def arima_features(df_list, stock_names, what_is_predicted, d_upper_bound, pq_upper_bound):
    forcast_list = []
    for k in range(len(df_list)):
        best_aic = np.inf 
        best_order = None
        best_model = None
        
        curr_df = df_list[k].copy()
        curr_df.set_index('Date', inplace = True, drop = True)
        curr_df = curr_df[what_is_predicted]

        pq_lower_bound = 0
        d_lower_bound = 0
        
        for i in range(pq_lower_bound, pq_upper_bound):
            for d in range(d_lower_bound, d_upper_bound):
                for j in range(pq_lower_bound, pq_upper_bound):
                    try:
                        
                        curr_model = smt.ARIMA(curr_df, order = (i, d, j))\
                        .fit(method = 'mle', trend = 'nc')
                        curr_aic = curr_model.aic
                        if curr_aic < best_aic:
                            best_aic = curr_aic
                            best_order = (i, d, j)
                            best_model = curr_model
                    except:
                        continue
        if best_model:        
            _ = tsplot(best_model.resid, stock_names[k], lags=None, figsize=(10, 8), style='bmh') 
            forcast_list.append(final_prediction(best_model, best_order, best_aic,\
                                                 df_list[k], stock_names[k], what_is_predicted)) 
    return forcast_list


    # this function is written following the example in
    # http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016
def tsplot(y, stock_name, lags = None, figsize = (10, 8), style = 'bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize = figsize)

        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan = 2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax = ts_ax)
        ts_title = 'Time Series Analysis Plots ' + stock_name 
        ts_ax.set_title(ts_title)
        smt.graphics.plot_acf(y, lags = lags, ax = acf_ax, alpha = 0.5)
        smt.graphics.plot_pacf(y, lags = lags, ax = pacf_ax, alpha = 0.5)
        sm.qqplot(y, line = 's', ax = qq_ax)
        qq_title = 'QQ Plot'#+ ' for Stock ' +stock_name
        qq_ax.set_title(qq_title)        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot = pp_ax)

        plt.tight_layout()

        
    # this function is written following the example in
    # http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016
def final_prediction(best_model, best_order, best_aic, df, stock_name, what_is_predicted):
    df_copy = df.copy() 
    # Create a 21 day forecast of SPY returns with 95%, 99% CI
    n_steps = 21

    f, err95, ci95 = best_model.forecast(steps = n_steps) # 95% CI
    _, err99, ci99 = best_model.forecast(steps = n_steps, alpha = 0.01) # 99% CI

    # i should pass original data closing price stuff
    idx = pd.date_range(df_copy['Date'].iloc[-1], periods = n_steps, freq = 'D')
    fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index = idx, columns = ['forecast', 'lower_ci_95', 'upper_ci_95'])
    fc_99 = pd.DataFrame(np.column_stack([ci99]), index = idx, columns = ['lower_ci_99', 'upper_ci_99'])
    fc_all = fc_95.combine_first(fc_99)
    #print(fc_all)
    
    ###############
    # Plot 21 day forecast for SPY returns
    plt.style.use('bmh')
    fig = plt.figure(figsize = (9,7))
    ax = plt.gca()

    df_copy.set_index('Date', inplace = True, drop = True)
    ts = df_copy[what_is_predicted].copy()
    ts.plot(ax = ax, label = what_is_predicted)
    
    pred = best_model.predict() 
    if best_order[1]: #differencing occurs
        pred += ts.shift(1)
    pred.plot(ax = ax, style = 'r-', label = 'In-sample prediction')

    styles = ['b-', '0.2', '0.75', '0.2', '0.75']
    fc_all.plot(ax = ax, style = styles)
    plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color = 'gray', alpha = 0.7)
    plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color = 'gray', alpha = 0.2)
    plt.title('{} Day Stock {} {} Forecast \n ARIMA{}'.format(n_steps, stock_name, what_is_predicted, best_order))
    plt.legend(loc = 'best', fontsize = 10)
    
    return fc_all
       
   #############################################################################################
   
# interval length (last feature) is how often we compute closing prices in minutes.
interval_length_one_minute = 1
interval_lenth_one_day = None
df_list_one_minute = find_relevant_info(df, stock_names, interval_length_one_minute)
df_list_one_day = find_relevant_info(df, stock_names, interval_lenth_one_day)
based_on_column_name = 'Close'

def concatenate_forcast(df_list, forcast_list):
    # do some stuff with confidense intervals
    df_list.append(forcast_list, ignore_index = True)
    
def ratio_then_vol_then_arima(df_list, interval_length):
    # period length is how often we use closing prices/ratios to compute volatility.
    # if period_length in minutes is None, then period_length_in_days is used.
    period_length_in_days = 1
    period_length_in_minutes = None # None or 60 minutes are preferred 
    find_rates(df_list, based_on_column_name)
    vol_df_list = find_volatility(df_list, period_length_in_minutes, period_length_in_days, interval_length)
    
    what_is_predicted = "Volatility"
    print(arima_features(vol_df_list, stock_names, what_is_predicted, 2, 5))

    
ratio_then_vol_then_arima(df_list_one_minute, interval_length_one_minute)    

def ratio_then_arima_then_vol(df_list, interval_length):
    find_rates(df_list, based_on_column_name)
    what_is_predicted = "Volatility"
    forcast_list = arima_features(df_list, stock_names, what_is_predicted, 2, 5)
    # the rest needs fixing, doesn't work
    period_length_in_days = 21 # monthly volatility
    period_length_in_minutes = None  
    vol_df_list = find_volatility(df_list, period_length_in_minutes, period_length_in_days, interval_length)
    
#ratio_then_arima_then_vol(df_list_one_day, interval_length_one_day)
    
def arima_then_ratio_then_vol(df_list):
    what_is_predicted = "Volatility"
    forcast_list = arima_features(df_list, stock_names, what_is_predicted, 2, 5)
    # complete the rest
    
    
#arima_then_ratio_then_vol(df_list_one_day)
