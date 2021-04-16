# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 18:10:28 2021

@author: AdamJackson
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:35:34 2021

@author: AdamJackson
"""

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
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve, plot_precision_recall_curve
from sklearn.calibration import calibration_curve
import json
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import f1_score
import shap

# %%

file_path = os.getcwd() + r'/raw_data.csv'
raw_data = pd.read_csv(file_path)

income_file_path = os.getcwd() + r'/income_by_state.csv'
income_data = pd.read_csv(income_file_path)

all_cols_data = pd.merge(raw_data, income_data, left_on='state', right_on='State', how='left')

gdp_file_path = os.getcwd() + r'/GDP_year_state.csv'
gdp_data = pd.read_csv(gdp_file_path)

gdp_col = {}
for i in range(len(gdp_data['GeoName'])):
    year_col = {}
    for j in range(1,15):
        year_col[gdp_data.columns[j]] = gdp_data[gdp_data.columns[j]][i]
    gdp_col[gdp_data[gdp_data.columns[0]][i]] = year_col

gdp_years = pd.DataFrame(gdp_col.values())
gdp_col_df = pd.DataFrame(gdp_col)

states_list = []
for i in gdp_col.keys():
    for j in gdp_col[i]:
        states_list.append(i)
        states_list.append(j)
        states_list.append(gdp_col[i][j])

states_for_df = states_list[0::3]
years_for_df = states_list[1::3]
gdp_for_df = states_list[2::3]

gdp_df = pd.DataFrame({'state':states_for_df, 'year':years_for_df, 'gdp':gdp_for_df})
gdp_df['year'] = pd.to_datetime(gdp_df['year'])
gdp_df['year'] = gdp_df['year'].dt.to_period('Y')

all_cols_data['year'] = pd.to_datetime(all_cols_data['transaction_date'])
all_cols_data['year'] = all_cols_data['year'].dt.to_period('Y')

final_df = pd.merge(all_cols_data, gdp_df, left_on=['state','year'], right_on=['state','year'], how='left')

final_features = ['start_balance', 'average_monthly_transactions_to_date', 'withdrawal','deposit', 'CPIAUCSL', 'UMCSENT', 'DGS1MO', 'gdp','HouseholdIncome', 'number_of_months_with_positive_transactions', 'number_of_months_with_negative_transactions', 'tenure', 'UNRATE','Age', 'amount', 'DFF', 'months_of_inactivity', 'running_account_total','transaction_date','last_transaction']
preprocessed_df = final_df[final_features]

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
        
model = GradientBoostingClassifier()

model.fit(X_train, y_train)
y_hat_train = model.predict(X_train)
train_acc = accuracy_score(y_train, y_hat_train)

y_hat_val = model.predict(X_val)
y_hat_val_proba = model.predict_proba(X_val)
val_acc = accuracy_score(y_val, y_hat_val)

feature_importances = model.feature_importances_


# %%

importances = list(zip(X_val.columns, feature_importances))
importances_df = pd.DataFrame(importances, columns=['feature', 'importance'])


# %%

brier = brier_score_loss(y_val, y_hat_val_proba[:,1])

# %%
# =============================================================================
# Plotting confusion matrix
# =============================================================================
#plt.rcParams.update({'font.size': 22})
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

# %%

'''from sklearn import metrics
preds = model.predict_proba(X_val)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_val,preds[:,1])
print (thresholds2)

accuracy_ls = []
for thres in thresholds2:
    y_pred = np.where(preds[:,1]>thres,1,0)
    accuracy_ls.append(metrics.accuracy_score(y_val, y_pred, normalize=True))


'''
# %%
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# %%

shap.force_plot(explainer.expected_value, shap_values[0,:], X_val.iloc[0,:])

# %%
shap.dependence_plot('months_of_inactivity', shap_values, X_val)

# %%

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_val)

# %%

file_path = os.getcwd() + r'/submission_janus.csv'
submission_data = pd.read_csv(file_path)
submission_data['pred_churn'] = model.predict_proba(scaler.transform(test_data))[:,1]

submission_data.to_csv(r'submission_janus.csv',index=False)

# %%

fig, ax = plt.subplots()
submission_data['pred_churn'].plot.hist(ax=ax, bins=100)

# %%

file_path = os.getcwd() + r'/submission_janus_first.csv'
submission_data = pd.read_csv(file_path)
submission_data['pred_churn'] = model.predict_proba(scaler.transform(test_data))[:,1]

submission_data.to_csv(r'submission_janus.csv',index=False)
