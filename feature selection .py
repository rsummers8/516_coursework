# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:01:54 2024

@author: russe
"""

#%%
# Elastic net method for normal data

from sklearn.linear_model import ElasticNetCV

# Create an instance of ElasticNetCV
elastic_net = ElasticNetCV(cv=10, random_state=42) # set cv = 10 as this is standard practise and relatively small dataset

# Fit the model on the training data
elastic_net.fit(X_train, y_train)

# Get the indices of selected features
selected_feature_indices = [i for i, coef in enumerate(elastic_net.coef_) if coef != 0]

# Get the names of selected features
selected_features = X_train.columns[selected_feature_indices]

# Print the selected features
print("Selected Features:")
print(selected_features)

#%%
# Filter X_train and X_test to include only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

#%%
# do the same for the outliers removed df

# Create an instance of ElasticNetCV
elastic_net = ElasticNetCV(cv=10, random_state=42) # set cv = 10 as this is standard practise and relatively small dataset

# Fit the model on the training data
elastic_net.fit(X_train_treated, y_train_treated)

# Get the indices of selected features
treated_selected_feature_indices = [i for i, coef in enumerate(elastic_net.coef_) if coef != 0]

# Get the names of selected features
treated_selected_features = X_train_treated.columns[treated_selected_feature_indices]

# Print the selected features
print("Treated Data - Selected Features:")
print(treated_selected_features)

#%%
# same for outliers removed data
# Filter X_train and X_test to include only selected features
X_train_treated_selected = X_train_treated[treated_selected_features]
X_test_treated_selected = X_test_treated[treated_selected_features]

#%%