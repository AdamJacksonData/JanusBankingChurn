# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:12:58 2021

@author: AdamJackson
"""

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve, plot_precision_recall_curve
from sklearn.calibration import calibration_curve
import json
import requests
from bs4 import BeautifulSoup

# %%

file_path = os.getcwd() + r'/Aggregated_data.csv'
raw_data = pd.read_csv(file_path)

Fed_Funds = pd.read_csv(os.getcwd() + r"/Federal_Funds.csv",parse_dates=['DATE'], index_col='DATE')
Treasury = pd.read_csv(os.getcwd() + r"/Treasury_Maturity_1M.csv",parse_dates=['DATE'], index_col='DATE')
UMC_Sent = pd.read_csv(os.getcwd() + r"/UMC_Sentiment.csv",parse_dates=['DATE'], index_col='DATE')
Unemp = pd.read_csv(os.getcwd() + r"/Unemployment.csv",parse_dates=['DATE'], index_col='DATE')

#&&
Fed_Funds = Fed_Funds.resample('M').mean()
Fed_Funds = Fed_Funds.reset_index()
Fed_Funds['DATE'] = Fed_Funds['DATE'].dt.to_period('M')

Treasury.drop(Treasury.loc[Treasury['DGS1MO']=='.'].index, inplace=True)
Treasury['DGS1MO'] = Treasury['DGS1MO'].astype(float)
Treasury = Treasury.resample('M').mean()
Treasury = Treasury.reset_index()
Treasury['DATE'] = Treasury['DATE'].dt.to_period('M')

UMC_Sent = UMC_Sent.reset_index()
UMC_Sent['DATE'] = UMC_Sent['DATE'].dt.to_period('M')

Unemp = Unemp.reset_index()
Unemp['DATE'] = Unemp['DATE'].dt.to_period('M')
 
Extern = pd.merge(Fed_Funds,Treasury, on='DATE', how = 'outer' )
Extern = pd.merge(Extern,UMC_Sent, on='DATE', how = 'outer' )
Extern = pd.merge(Extern,Unemp, on='DATE', how = 'outer' )

raw_data['transaction_date'] = raw_data['transaction_date'].values.astype('datetime64[M]')
raw_data['transaction_date'] = raw_data['transaction_date'].dt.to_period('M')
raw_data = raw_data.join(Extern.set_index('DATE'), on ='transaction_date', how = 'left')

# %%
raw_data['end_date'] = pd.to_datetime('2020-05-31')
raw_data['dob'] = pd.to_datetime(raw_data['dob'])
raw_data['Age'] = (raw_data['end_date'] - raw_data['dob'])/np.timedelta64(1,'Y')
raw_data['Age'] = raw_data['Age'].apply(np.floor)


# %%

r = requests.get(r'https://inkplant.com/code/state-latitudes-longitudes')

soup = BeautifulSoup(r.text, 'html.parser')

# %%
tag        = 'table'
attributes = {'class':'table table-hover'}
table_soup = soup.find(tag, attributes)

table_data = []
for row in table_soup.find_all('tr'):
    row_text = [e.text.strip() for e in row.find_all('td')]
    table_data.append(row_text)
    
table_cols = table_data[0]
table_content = table_data[1:]

location_df = pd.DataFrame(table_content, columns=table_cols)

preprocessed_df = raw_data.drop(columns=['customer_id', 'deposit', 'withdrawal', 'dob', 'creation_date', 'account_id', 'end_date'])


preprocessed_df = pd.merge(preprocessed_df, location_df, left_on='state', right_on='State', how='left')


#dummies = pd.get_dummies(preprocessed_df['state'], drop_first=False)
#states = sorted(preprocessed_df['state'].unique())

#for i in range(len(states)):
#    preprocessed_df[states[i]] = dummies[dummies.columns[i]]

preprocessed_df = preprocessed_df.drop(columns=['state','State'])
# %%

test_data = preprocessed_df[preprocessed_df['transaction_date']=='2020-05']
train_val_data = preprocessed_df[preprocessed_df['transaction_date']!='2020-05']

test_data.drop(columns=['transaction_date','last_transaction'],inplace=True)
train_val_data['churn'] = train_val_data['last_transaction'].map({True:1,False:0})
train_val_data.drop(columns=['transaction_date','last_transaction'],inplace=True)

#churners = train_val_data[train_val_data['churn']==1]
#matched_non_churners = train_val_data[train_val_data['churn']==0].sample(98764, random_state=123)

#train_val_data_matched = churners.append(matched_non_churners)
np.nan_to_num(train_val_data)
# %%

train_data, val_data = train_test_split(train_val_data, test_size=0.2, stratify=train_val_data['churn'], shuffle=True, random_state=123)

X_train = train_data.drop(columns='churn')
y_train = train_data['churn']
X_val = val_data.drop(columns='churn')
y_val = val_data['churn']

# %%

scaler=StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val),columns=X_val.columns)

# %%
# Bagged Tree/Classifier
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()

model.fit(X_train, y_train)
y_hat_train = model.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)

y_hat_val = model.predict(X_val)
y_hat_val_proba = model.predict_proba(X_val)
val_acc = accuracy_score(y_val, y_hat_val)

# %%
# =============================================================================
# Plotting confusion matrix
# =============================================================================
plt.rcParams.update({'font.size': 22})
plot_confusion_matrix(model, X_val, y_val)

# %%
# =============================================================================
# Plotting ROC curve
# =============================================================================

confusion = confusion_matrix(y_val, y_hat_val)
TPR = confusion[1][1]/(confusion[1][1]+confusion[1][0])
TNR = confusion[0][0]/(confusion[0][0]+confusion[0][1])

fpr, tpr, thesholds = roc_curve(y_val, y_hat_val_proba[:,1])
auc = roc_auc_score(y_val, y_hat_val_proba[:,1])
plot_roc_curve(model, X_val, y_val)


# %%
# =============================================================================
# Plotting PR curve
# =============================================================================

plot_precision_recall_curve(model, X_val, y_val)

# %%
# =============================================================================
# Plotting calibration curve
# =============================================================================

fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_hat_val_proba[:,1], n_bins=10)

x_perf = np.linspace(0.05,0.9,10)
y_perf = np.linspace(0.05,0.9,10)
fig, ax = plt.subplots()
ax.plot(mean_predicted_value, fraction_of_positives, marker='o')
ax.plot(x_perf, y_perf, c='red', linestyle='--')
ax.set_xlabel('Predictions')
ax.set_ylabel('Proportion of positive observations')
ax.grid()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title('Calibration Curve')


# %%


from sklearn.tree import export_graphviz

sub_tree = model.estimators_[0, 0]

from pydotplus import graph_from_dot_data
from IPython.display import Image
dot_data = export_graphviz(
    sub_tree,
    out_file=None, filled=True, rounded=True,
    special_characters=True,
    proportion=False, impurity=False, # enable them if you want
)
graph = graph_from_dot_data(dot_data)
Image(graph.create_png())