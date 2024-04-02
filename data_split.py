# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:44:48 2024

@author: russe
"""
from sklearn.model_selection import train_test_split

seed = 5
np.random.seed(seed)

X, y = b_cancer.drop(columns=['Grade']), b_cancer['Grade'] 

# split data into train & test sets: 70% training & 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# confirm splitting
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#%%
#%%
# split no_outliers data under same parameters
# ensure no names conflict 
seed = 5 
np.random.seed(seed)
X_treated, y_treated = b_cancer_treated.drop(columns=['Grade']), b_cancer_treated['Grade']

# Split data into train and test set (70% training, 30% testing)
X_train_treated, X_test_treated, y_train_treated, y_test_treated = train_test_split(X_treated, y_treated, test_size=0.3, random_state=1)

# Confirm splitting
print(X_train_treated.shape, X_test_treated.shape, y_train_treated.shape, y_test_treated.shape)
#%%