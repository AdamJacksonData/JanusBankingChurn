# %%
# Imports

import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def date_split(date_str):
    year  = date_str[0:4]
    month = date_str[5:7]
    day   = date_str[8:10]

    return {'year':year, 'month':month, 'day':day}

def calculate_age(birth_date):
    birth = date_split(birth_date)
    curr  = date_split('2020-05-31')

    birth_num = int(birth['month']+ birth['day'])
    curr_num  = int(curr['month'] + curr['day'])
    
    if birth_num <= curr_num:
        age = int(curr['year']) - int(birth['year'])
    else:
        age = int(curr['year']) - int(birth['year']) - 1
    return age

# %%
# Merge Last Transaction Date to customer

last_path = r'last_dates.csv'
last_date = pd.read_csv(last_path)

cust_path = r'customers_tm1_e.csv'
customers = pd.read_csv(cust_path)

last_date.drop(columns='Unnamed: 0', inplace=True)

customer_info = pd.merge(last_date, customers, on='customer_id', how='left')

# %% 
# Add Churn column

for i in range(len(customer_info)):
    if customer_info.loc[i, 'most_recent_trans'][:7] == '2020-05':
        customer_info.loc[i, 'churn'] = 1
    else:
        customer_info.loc[i, 'churn'] = 0
    
# %%
# Add Age Column 

customer_info['Age'] = customer_info['dob'].apply(calculate_age)

# %%
# Starting Balance for Churners and Non-Churners. 

# Set boundaries for starting Balance.
for i in range(len(customer_info)):
    if customer_info.loc[i, 'start_balance'] > 1000000:
        customer_info.loc[i, 'start_balance'] = 0
    if customer_info.loc[i, 'start_balance'] < 0:
        customer_info.loc[i, 'start_balance'] = 0
      
# Plot the Starting Balances proportions for churners and non-churners.
churned  = customer_info[customer_info['churn'] == 1]
no_churn = customer_info[customer_info['churn'] == 0]

churn_bal    = churned['start_balance']
no_churn_bal = no_churn['start_balance']

fig, ax = plt.subplots(figsize=(12,6))

ax.hist([churn_bal, no_churn_bal], density=True, bins=20)
ax.legend(['churners', 'non_churners'])
ax.set_ylabel('Density')
ax.set_xlabel('Starting Balance')
    
# %%
# Churn Rate by age

age_set_list = list(set(customer_info['Age']))

ages_list = []
age_rates = []

for i in age_set_list: 
    age_mask = customer_info['Age'] == i
    
    customers_by_age = customer_info[age_mask]
    
    churn_mask    = customers_by_age['churn'] == 1
    no_churn_mask = customers_by_age['churn'] == 0
    
    churners     = len(customers_by_age[churn_mask])
    non_churners = len(customers_by_age[no_churn_mask])
                       
    proportion   = churners/(non_churners+churners)
    
    ages_list.append(i)
    age_rates.append(proportion)
    
# %%

fig, ax = plt.subplots(figsize=(12,6))

ax.bar(ages_list, age_rates)
ax.set_xlabel('Age')
ax.set_ylabel('Churn rate')
    

    
    
    
    
    
    
    
    
    

















