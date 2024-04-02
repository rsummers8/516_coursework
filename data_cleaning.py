# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:50:51 2024

@author: russe
"""
#%%
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

# load the dataset 

b_cancer = pd.read_csv('Dataset _01.csv')

# data cleaning 

# inspect data

b_cancer.shape

# Inspect: number of samples and number of samples
print(str("dataset has ") + str(b_cancer.shape[0])+str(' samples / instances & ')+str(b_cancer.shape[1])+ str(' features'))
b_cancer.head(10)
b_cancer.columns
#%%
#%%
# count number of missing values for each feature

b_cancer.isna().sum()

# no missing values 
#%%
#%%
# check for uplicate rows 

duplicate_rows = b_cancer[b_cancer.duplicated()]
print(duplicate_rows) # empty 
#%%
#%%
#check data types 

print(b_cancer.dtypes) #int64 is sufficient for binary classification
#%%