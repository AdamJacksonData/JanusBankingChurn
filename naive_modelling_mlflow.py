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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import argparse
from sklearn.tree import plot_tree

# %%
tracking_uri = r'file:C:/Users/AdamJackson/junkdata/mlruns'

parser = argparse.ArgumentParser('Banking Churn - GBC')
parser.add_argument('--n', type=int, default=1, help='Number of experiments to run')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')

# Parser
args, _ = parser.parse_known_args()
n_experiments = args.n
lr = args.lr

# %%

file_path = os.getcwd() + r'/Aggregated_data.csv'
raw_data = pd.read_csv(file_path)

# %%
raw_data['end_date'] = pd.to_datetime('2020-05-31')
raw_data['dob'] = pd.to_datetime(raw_data['dob'])
raw_data['Age'] = (raw_data['end_date'] - raw_data['dob'])/np.timedelta64(1,'Y')
raw_data['Age'] = raw_data['Age'].apply(np.floor)

preprocessed_df = raw_data.drop(columns=['customer_id', 'deposit', 'withdrawal', 'dob', 'creation_date', 'account_id', 'end_date'])
#preprocessed_df = preprocessed_df.drop(columns='transaction_date')
preprocessed_df = preprocessed_df.drop(columns='state')

# %%

test_data = preprocessed_df[preprocessed_df['transaction_date']=='2020-05']
train_val_data = preprocessed_df[preprocessed_df['transaction_date']!='2020-05']

test_data.drop(columns=['transaction_date','last_transaction'],inplace=True)
train_val_data['churn'] = train_val_data['last_transaction'].map({True:1,False:0})
train_val_data.drop(columns=['transaction_date','last_transaction'],inplace=True)

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

mlflow.set_tracking_uri(tracking_uri) # if we don't specify, logs are written in cwd
mlflow.set_experiment('Banking Churn - GBC')

for i in range(n_experiments):
    with mlflow.start_run():
        
        md = np.random.randint(1, 10)
        msl = np.random.randint(1, 25)
        model = GradientBoostingClassifier(learning_rate=lr, max_depth=md, min_samples_leaf=msl)

        model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_hat_train)
    
        y_hat_val = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_hat_val)
        
        mlflow.log_param('max_depth', md)
        mlflow.log_param('min_samples_leaf', msl)
        mlflow.log_param('learning_rate', lr)
        mlflow.log_metric('train_acc',train_acc)
        mlflow.log_metric('val_acc',val_acc)
        mlflow.sklearn.log_model(model, 'BankingChurnGBC')


# %%

'''
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
'''
# %%










