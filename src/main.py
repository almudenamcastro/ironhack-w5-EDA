# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np
import statsmodels.api as sm
import sys
# append the path of the parent directory
sys.path.append("..")
from lib.functions import *

# load the raw data
clients_raw = pd.read_csv('../data/raw/df_final_demo.txt')
events_1_raw = pd.read_csv('../data/raw/df_final_web_data_pt_1.txt')
events_2_raw = pd.read_csv('../data/raw/df_final_web_data_pt_2.txt')
test_raw = pd.read_csv('../data/raw/df_final_experiment_clients.txt')

# clean data: 

# clients: 
clients = clients_raw.copy()
clients.dropna(inplace=True) #drop nulls
int_cols = ['clnt_tenure_yr', 'clnt_tenure_mnth', 'num_accts', 'calls_6_mnth', 'logons_6_mnth']
clients[int_cols] = clients[int_cols].astype('int64') # convert to int
clients = clients[clients['clnt_age'] > clients['clnt_tenure_yr']] # filter impossible data

# events: 
events = pd.concat([events_1_raw, events_2_raw]) #merge two datasets
events.drop_duplicates(inplace=True) # drop duplicates
events['date_time'] = pd.to_datetime(events['date_time']) #convert to datetime

# test:
test = test_raw.copy()
test = test.dropna()

# test datasets: 

# EVENT METRICS
# Duration and step metrics. 
test_events = events.copy()

# turn the step column into a numeric value. 
test_events['step'] = test_events['process_step'].replace({'start': '0', 'step_1': '1', 'step_2': '2', 'step_3': '3', 'confirm': '4'}).astype(int)

# shift the table to calculate the time spent in each step and the next step.
# we need to sort by visit_id and date_time first: 
test_events = test_events.sort_values(['client_id', 'visit_id', 'date_time']).reset_index()
test_events[['next_time', 'next_step']] = test_events[['visit_id', 'date_time','step']].groupby('visit_id').shift(-1)
test_events = test_events.rename(columns={'date_time':'time'})
test_events['next_step'] = test_events['next_step'].apply(lambda x: int(x) if pd.notnull(x) else np.nan)

# get the time spent on each step
test_events['time_inc'] = (test_events['next_time']- test_events['time']).dt.total_seconds()

# get the 'step_inc' (difference between the next step and the current step)
test_events['step_inc'] = test_events['next_step'] - test_events['step']

# get the 'conversion' (1 if the step is the last step, 0 otherwise)
test_events['conversion'] = test_events['step'].apply(lambda x: 1 if x == 4 else 0)

# get the 'return'
test_events['return'] = (test_events['step_inc'] < 0).astype(int)
test_events['refresh'] = (test_events['step_inc'] == 0).astype(int)


# change columns order to make them easier to read. 
test_events = test_events[['client_id', 'visitor_id', 'visit_id', 'time', 'next_time', 'time_inc', 'process_step', 'step', 'next_step', 'step_inc', 'conversion', 'return', 'refresh']]

# METRICS BY CLIENT
# We will now add the client demographic info and calculate metrics per client. 
test_clients = pd.merge(test, clients, on='client_id', how='inner')

# we will group the events by client_id and calculate: 
# * total visits, 
# * total events, 
# * total conversions,
# * total time
# * last step, 

test_clients = pd.merge(test_clients, test_events.
                        groupby('client_id').agg({'visit_id': 'count', 'time': 'size','conversion': 'sum', 'time_inc': 'sum', 'step': 'max'}).
                        reset_index().
                        rename(columns = {'visit_id': 'total_visits', 'time_inc': 'total_time', 'conversion': 'total_conversions', 'step': 'last_step', 'time': 'total_events'}), 
                        on='client_id', how='left')

# add converstion to test df  
# the value will be 1 if the user completed the process and 0 if they did not.
test_clients['conversion'] = (test_clients['total_conversions'] > 0).astype(int)

# get the average time spent in each step per client
test_clients = pd.merge(test_clients, test_events.groupby('client_id')['time_inc'].mean().reset_index(name='avg_step_time'), on='client_id', how='left')

# get the number of errors (step_inc < 0) per client 
test_clients = pd.merge(test_clients, test_events[test_events['step_inc'] < 0].groupby('client_id').size().reset_index(name='total_returns'), on='client_id', how='left')
# we have deteremined that null values in return_count correspond to users that didn't have any erros. 
test_clients.total_returns = test_clients.total_returns.fillna(0).astype(int)

# we can calculate the return rate per user. 
test_clients['return_rate'] = test_clients['total_returns']/test_clients['total_events']

# Let's check the null values in the 'avg_step_time' column.
test_clients[test_clients.avg_step_time.isnull()]['last_step'].value_counts()

# these values indicate that the user didn't have any other events in during the same visit. 
# this could be a sign of two different phenomena: 
# - if the user landed directly on the confirm page, this could mean that they are using an API to submit the form.
# - if the user landed on step 1 and didn't move any further, this could indicate a high bouce rate. 

# we will remove the values that indicate the use of an API. 
test_clients = test_clients[(~(test_clients.avg_step_time.isnull()) | (test_clients['last_step'] == 0))]

# We will leave the rest of the nan values for now to calculate the bouce rate.

# Add the variation in the events dataframe
test_events = pd.merge(test_events, test_clients[['client_id', 'Variation']], on='client_id', how='inner')

# add the user_converted column
test_events = pd.merge(test_events, test_events.groupby('client_id')['conversion'].sum().reset_index(name = 'user_converted'),
                        on='client_id', how='left')

test_events['user_converted'] = (test_events['user_converted']>0).astype(int)

test_clients.to_csv('../data/cleaned/test_clients.csv', index=False)
test_events.to_csv('../data/cleaned/test_events.csv', index=False)