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

customer_raw['state'].replace(to_replace={'NY':'New York','TX':'Texas','CALIFORNIA':'California','UNK':'Nebraska','MASS':'Massachusetts'},inplace=True)

combined_df = pd.merge(customer_raw, transaction_raw, on='customer_id', how='left')


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

