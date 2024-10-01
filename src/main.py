# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np
import statsmodels.api as sm

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
clients = clients[clients['clnt_age']>clients['clnt_tenure_yr']] # filter impossible data

# events: 
events = pd.concat([events_1_raw, events_2_raw]) #merge two datasets
events.drop_duplicates(inplace=True) # drop duplicates
events['date_time'] = pd.to_datetime(events['date_time']) #convert to datetime

# test:
test = test_raw.copy()
test = test.dropna()

# test metrics data set: 
