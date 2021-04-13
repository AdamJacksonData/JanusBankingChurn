# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 07:31:31 2021

@author: AdamJackson

Exploration of datasets
"""
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%

customers_file_path = os.getcwd() + r'/customers_tm1_e.csv'
transactions_file_path = os.getcwd() + r'/transactions_tm1_e.csv'
customer_raw = pd.read_csv(customers_file_path)
transaction_raw = pd.read_csv(transactions_file_path)

# %%
# =============================================================================
# Load macroeconomics data
# =============================================================================
start_date = pd.to_datetime('2007-01-31')
end_date   = pd.to_datetime('2020-04-30')

# University of Michigan Consumer Sentiment Index
umcsent_df = pd.read_csv(r'./csvs/UMCSENT.csv')
umcsent_df['DATE'] = pd.to_datetime(umcsent_df['DATE'])
umcsent_df.set_index('DATE', inplace=True)
umcsent_df = umcsent_df.resample('M').mean()
umcsent_df = umcsent_df[start_date:end_date]

# Unemployment Rate
unrate_df = pd.read_csv(r'./csvs/UNRATE.csv')
unrate_df['DATE'] = pd.to_datetime(unrate_df['DATE'])
unrate_df.set_index('DATE', inplace=True)
unrate_df = unrate_df.resample('M').mean()
unrate_df = unrate_df[start_date:end_date]

#Effective Federal Funds Rate
dff_df = pd.read_csv(r'./csvs/DFF.csv')
dff_df['DATE'] = pd.to_datetime(dff_df['DATE'])
dff_df.set_index('DATE', inplace=True)
dff_df = dff_df.resample('M').mean()
dff_df = dff_df[start_date:end_date]

# Gross Domestic Product
gdp_df = pd.read_csv(r'./csvs/GDP.csv')
gdp_df['DATE'] = pd.to_datetime(gdp_df['DATE'])
gdp_df.set_index('DATE', inplace=True)
gdp_df = gdp_df.resample('M').mean()
gdp_df = gdp_df.interpolate()
gdp_df = gdp_df[start_date:end_date]

# Inflation rate
t10yie_df = pd.read_csv(r'./csvs/T10YIE.csv')
t10yie_df['DATE'] = pd.to_datetime(t10yie_df['DATE'])
t10yie_df.set_index('DATE', inplace=True)
t10yie_df = t10yie_df[t10yie_df['T10YIE'] != '.']
t10yie_df['T10YIE'] = pd.to_numeric(t10yie_df['T10YIE'])
t10yie_df = t10yie_df.resample('M').mean()
t10yie_df = t10yie_df[start_date:end_date]

# %%
customer_raw['state'].replace(to_replace={'NY':'New York','TX':'Texas','CALIFORNIA':'California','UNK':'Nebraska','MASS':'Massachusetts', 'District of Columbia':'Washington'},inplace=True)
combined_df = pd.merge(customer_raw, transaction_raw, on='customer_id', how='left')

# dtype to date
for col in ['transaction_date', 'creation_date', 'dob', 'date']:
    combined_df[col] = pd.to_datetime(combined_df[col])#.dt.strftime('%Y-%m-%d')

# %%
combined_df.drop(combined_df[combined_df['state']=='Australia'].index,inplace=True)
combined_df.drop(combined_df[combined_df['state']=='-999'].index,inplace=True)

combined_df.sort_values(by=['customer_id', 'transaction_date'], inplace=True, ascending=True)


# %%
# =============================================================================
# Most recent transactions
# =============================================================================
combined_df['last_transaction'] = ~combined_df.duplicated(subset=['customer_id'], keep='last')
combined_df.sort_values(by=['customer_id', 'transaction_date'], inplace=True)




# %%
neg_sb_df = combined_df[combined_df['start_balance']<0]
large_sb_df = combined_df[combined_df['start_balance']>2.5e8]

rows_to_drop = neg_sb_df['customer_id'].unique()[0]
large_rows_to_drop = large_sb_df['customer_id'].unique()

combined_df.drop(combined_df[combined_df['customer_id']==rows_to_drop].index, inplace=True)
combined_df.drop(combined_df[combined_df['start_balance']>2.5e8].index, inplace=True)

# %%

high_amount = combined_df[combined_df['amount']>2.5e8]
combined_df.drop(combined_df[combined_df['amount']>2.5e8].index, inplace=True)
low_amount = combined_df[combined_df['amount']<-1e8]
combined_df.drop(combined_df[combined_df['amount']<-1e8].index, inplace=True)
df_stats = combined_df.describe()


# %%
# =============================================================================
# Running account balance for each customer
# =============================================================================
combined_df['running_account_total'] = combined_df.groupby(['customer_id'])['amount'].cumsum() + combined_df['start_balance']


# %%
# =============================================================================
# plotting
# =============================================================================

df_sample = combined_df.sample(frac=0.001)

pd.plotting.scatter_matrix(df_sample.drop(columns='last_transaction'), c=df_sample['last_transaction'].map({True:'blue', False:'pink'}))


# %%
# =============================================================================
# portfolio analysis
# =============================================================================
customer_raw.drop(customer_raw[abs(customer_raw['start_balance'])>1e7].index, inplace=True)

temp_df = customer_raw[['start_balance', 'creation_date']]
temp_df['creation_date'] = pd.to_datetime(temp_df['creation_date'])
temp_df = temp_df.set_index('creation_date')
temp_df_m = temp_df.resample('M').sum()

port_df = combined_df.set_index('transaction_date')[['amount', 'running_account_total', 'last_transaction']]
port_df['amount'][port_df['last_transaction'] == True] = -port_df['running_account_total']
port_df_m = port_df[['amount']].resample('M').sum()

portfolio_df = pd.merge(temp_df_m, port_df_m, left_index=True, right_index=True)
portfolio_df.drop(index=pd.to_datetime('2020-05-31 00:00:00'), inplace=True)
portfolio_df['total_monthly_change'] = portfolio_df['start_balance'] + portfolio_df['amount']
portfolio_df['portfolio_balance']    = portfolio_df['total_monthly_change'].cumsum()
portfolio_df['zero'] = np.zeros((160,1))

portfolio_df['university_of_michigan_consumer_sentiment_index'] = umcsent_df
portfolio_df['unemployment_rate'] = unrate_df
portfolio_df['effective_federal_funds_rate'] = dff_df
portfolio_df['gdp'] = gdp_df
portfolio_df['inflation'] = t10yie_df

portfolio_df.rename({'start_balance':'monthly_income_from_account_openings', 'amount':'monthly_transactions_and_account_closures','total_monthly_change':'monthly_portfolio_balance_change'}, axis=1, inplace=True)
portfolio_df[['portfolio_balance', 'monthly_income_from_account_openings', 'monthly_transactions_and_account_closures', 'monthly_portfolio_balance_change', 'zero']].plot(xlabel='Date', ylabel='Portfolio balence in USD', style=['-','-','-','-','--'])

portfolio_df[['portfolio_balance']].plot(fontsize=20)
plt.xlabel('Date', fontsize=26)
plt.ylabel('Portfolio balence in USD (*1e7)', fontsize=26)
plt.legend(fontsize=22)

portfolio_df[['portfolio_balance', 'university_of_michigan_consumer_sentiment_index', 'unemployment_rate', 'effective_federal_funds_rate', 'gdp', 'inflation']].plot(xlabel='Date', subplots=True)


portfolio_df[['portfolio_balance', 'university_of_michigan_consumer_sentiment_index', 'unemployment_rate', 'effective_federal_funds_rate', 'gdp', 'inflation']].plot(subplots=True)
plt.xlabel('Date', fontsize=20)

