# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:25:22 2024

@author: russe
"""
#%%
# scale data    #don't run this cell to run wihout scaling 
# Import preprocessing library
from sklearn.preprocessing import StandardScaler

# Select numeric features excluding 'Subjects' and 'Grade'
numeric_features = b_cancer.drop(columns=['Subjects', 'Grade']).select_dtypes(include=['float64', 'int64'])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform only the numeric features
scaled_features = scaler.fit_transform(numeric_features)

# Create a DataFrame with scaled numeric features
scaled_df = pd.DataFrame(scaled_features, columns=numeric_features.columns)

# Concatenate scaled numeric features with 'Grade' column
b_cancer = pd.concat([scaled_df, b_cancer['Grade']], axis=1)
#%%

# Select numeric features excluding 'Subjects' and 'Grade'
numeric_features = b_cancer_treated.drop(columns=['Subjects', 'Grade']).select_dtypes(include=['float64', 'int64'])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform only the numeric features
scaled_features = scaler.fit_transform(numeric_features)

# Create a DataFrame with scaled numeric features
scaled_df = pd.DataFrame(scaled_features, columns=numeric_features.columns)

# Concatenate scaled numeric features with 'Grade' column
b_cancer_treated = pd.concat([scaled_df, b_cancer_treated['Grade']], axis=1)

#%%