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

customer_raw['state'].replace(to_replace={'NY':'New York','TX':'Texas','CALIFORNIA':'California','UNK':'Nebraska','MASS':'Massachusetts', 'District of Columbia':'Maryland'},inplace=True)

combined_df = pd.merge(customer_raw, transaction_raw, on='customer_id', how='left')

# dtype to date
for col in ['transaction_date', 'creation_date', 'dob', 'date']:
    combined_df[col] = pd.to_datetime(combined_df[col])

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

# =============================================================================
# Running account balance for each customer
# =============================================================================
combined_df['running_account_total'] = combined_df.groupby(['customer_id'])['amount'].cumsum() + combined_df['start_balance']


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
low_amount = combined_df[combined_df['running_account_total']<-1e8]
high_amount = combined_df[combined_df['running_account_total']>2.5e8]
combined_df.drop(combined_df[(combined_df['running_account_total']>2.5e8) | (combined_df['running_account_total']<-1e8)].index, inplace=True)
df_stats = combined_df.describe()


# %%
# =============================================================================
# plotting
# =============================================================================

df_sample = combined_df.sample(frac=0.001)

pd.plotting.scatter_matrix(df_sample.drop(columns='last_transaction'), c=df_sample['last_transaction'].map({True:'blue', False:'pink'}))

# %%
# dtype to period
for col in ['creation_date', 'dob', 'date']:
    combined_df[col] = pd.to_datetime(combined_df[col]).dt.to_period('D')
for col in ['transaction_date']:
    combined_df[col] = pd.to_datetime(combined_df[col]).dt.to_period('M')

# %%
# =============================================================================
# Code aggregating account amounts, deposits and withdrawals so there is only one entry per account per month.
# combo_agg_df is the updated version of combined_df
# =============================================================================
agg_df = combined_df[['customer_id', 'transaction_date', 'amount', 'deposit', 'withdrawal']].groupby(['customer_id','transaction_date']).sum()
agg_df.reset_index(inplace=True, level=['customer_id'])
agg_df.reset_index(inplace=True)

cols_to_keep = ['customer_id', 'dob', 'state', 'start_balance', 'creation_date', 'date',
       'account_id','transaction_date']

combo_agg_df = pd.merge(agg_df, combined_df[cols_to_keep], on=['customer_id','transaction_date'], how='inner')
combo_agg_df['last_transaction'] = ~combo_agg_df.duplicated(subset=['customer_id'], keep='last')
combo_agg_df.drop_duplicates(subset=['customer_id', 'transaction_date'], keep='last', inplace=True)

# %%
combo_agg_df['running_account_total'] = combo_agg_df.groupby(['customer_id'])['amount'].cumsum() + combo_agg_df['start_balance']

combo_agg_df['tenure'] = np.ceil((combo_agg_df['date'] - combo_agg_df['creation_date'])/np.timedelta64(1,'M'))

combo_agg_df.drop(columns='date', inplace=True)

# %%
# =============================================================================
# Code to check if any customers have multiple accounts - they don't
# =============================================================================
'''
for i in tqdm(combo_agg_df['customer_id'].unique()):
    if (len(combo_agg_df[combo_agg_df['customer_id']==i]['account_id'].unique())>1):
        print(i)
'''
# %%
# =============================================================================
# Filling unknown starting balances with the mean starting balance in the dataset
# =============================================================================
no_start_bal = combo_agg_df[combo_agg_df['start_balance'].isna()]
no_start_bal_ids = no_start_bal['customer_id'].unique()
combo_agg_df.fillna(value=(combo_agg_df['start_balance'].dropna()).mean(), inplace=True)
agg_df_stats = combo_agg_df.describe()

# %%
agg_df_sample = combo_agg_df.sample(frac=0.001)

pd.plotting.scatter_matrix(agg_df_sample.drop(columns='last_transaction'), c=agg_df_sample['last_transaction'].map({True:'blue', False:'pink'}))

# %%
# =============================================================================
# churn rate
# =============================================================================

#filter combined_df by last transactions made
churn_df = combo_agg_df[combo_agg_df['last_transaction']==True]
date_list = sorted(set(combo_agg_df['transaction_date']))

# making dataframe containing month, number of last transactions
lasttran_monthly = churn_df['transaction_date'].value_counts()
lasttran_monthly = pd.DataFrame(lasttran_monthly)
lasttran_monthly = lasttran_monthly.reset_index().sort_values(['index'])
lasttran_monthly = lasttran_monthly.rename(columns={'index':'date', 'transaction_date':'no_lt'}).set_index(['date'])
lasttran_monthly = lasttran_monthly.shift(periods=1).dropna().reset_index()

# finding total number of customers for each month
total_customers = []
for i in date_list:
    total_customer = combo_agg_df[combo_agg_df['transaction_date']==i]
    total_customers.append(len(total_customer['customer_id']))

total_customers.remove(total_customers[160])


# adding customer total and churn rate to dataframe
lasttran_monthly.insert(2,'total customers', total_customers)
lasttran_monthly['churn rate %'] = (lasttran_monthly['no_lt']/lasttran_monthly['total customers'])*100
churn_data_monthly = lasttran_monthly
churn_data_monthly['date'] = churn_data_monthly['date'].values.astype('datetime64[M]')

# plotting number of churns
fig, ax = plt.subplots()
ax.plot(churn_data_monthly['date'], churn_data_monthly['no_lt'], marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Customers Churned')
ax.set_title('Customer Churn per Month')

#plotting churn rate
fig, ax = plt.subplots()
ax.plot(churn_data_monthly['date'], churn_data_monthly['churn rate %'], marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Churn Rate %')
ax.set_title('Customer Churn Rate per Month')


