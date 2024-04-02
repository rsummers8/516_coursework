# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:34:28 2024

@author: russe
"""
#%%
# hyper parameter tuning of et non treated data model 
from sklearn.model_selection import GridSearchCV

seed = 6 
np.random.seed(seed)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=trained_et_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the training data
grid_search.fit(X_train_selected, y_train)

# Get the best parameters found by the grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator (model) found by the grid search
best_model = grid_search.best_estimator_

# Make predictions using the best model
best_model_predictions = best_model.predict(X_test_selected)

# Evaluate the best model's performance
best_accuracy = accuracy_score(y_test, best_model_predictions)
best_precision = precision_score(y_test, best_model_predictions)
best_recall = recall_score(y_test, best_model_predictions)
best_f1 = f1_score(y_test, best_model_predictions)
best_auc = roc_auc_score(y_test, best_model_predictions)

best_metrics = {
    'Accuracy': best_accuracy,
    'Precision': best_precision,
    'Recall': best_recall,
    'F1': best_f1,
    'AUC':best_auc}

best_metrics_df = pd.DataFrame([best_metrics])
best_metrics_df['Mean'] = best_metrics_df.mean(axis=1)

print(best_metrics_df)
#%%
rf_seed = 6 
np.random.seed(rf_seed)

# Define the parameter grid for hyperparameter tuning
rf_param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

# Initialize the GridSearchCV object
rf_grid_search = GridSearchCV(estimator=trained_rf_model, param_grid=rf_param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the training data
rf_grid_search.fit(X_train_selected, y_train)

# Get the best parameters found by the grid search
rf_best_params = rf_grid_search.best_params_
print("Best Parameters:", rf_best_params)

# Get the best estimator (model) found by the grid search
rf_best_model = rf_grid_search.best_estimator_

# Make predictions using the best model
rf_best_model_predictions = rf_best_model.predict(X_test_selected)

# Evaluate the best model's performance
rf_best_accuracy = accuracy_score(y_test, rf_best_model_predictions)
rf_best_precision = precision_score(y_test, rf_best_model_predictions)
rf_best_recall = recall_score(y_test, rf_best_model_predictions)
rf_best_f1 = f1_score(y_test, rf_best_model_predictions)
rf_best_auc = roc_auc_score(y_test, rf_best_model_predictions)

# Store the metrics in a dictionary
rf_best_metrics = {
    'Accuracy': rf_best_accuracy,
    'Precision': rf_best_precision,
    'Recall': rf_best_recall,
    'F1': rf_best_f1,
    'AUC': rf_best_auc
}

# Convert the dictionary to a DataFrame
rf_best_metrics_df = pd.DataFrame([rf_best_metrics])

# Calculate the mean of each row (metric)
rf_best_metrics_df['Mean'] = rf_best_metrics_df.mean(axis=1)

print(rf_best_metrics_df)

#%%

ab_seed = 6 
np.random.seed(ab_seed)

# Define the parameter grid for hyperparameter tuning
ab_param_grid = {
    'n_estimators': [50, 100, 150],  # Number of estimators
    'learning_rate': [0.1, 0.5, 1.0],  # Learning rate
    'algorithm': ['SAMME', 'SAMME.R']  # Algorithm to use for boosting
}

# Initialize the GridSearchCV object
ab_grid_search = GridSearchCV(estimator=trained_ab_model, param_grid=ab_param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the training data
ab_grid_search.fit(X_train_selected, y_train)

# Get the best parameters found by the grid search
ab_best_params = ab_grid_search.best_params_
print("Best Parameters:", ab_best_params)

# Get the best estimator (model) found by the grid search
ab_best_model = ab_grid_search.best_estimator_

# Make predictions using the best model
ab_best_model_predictions = ab_best_model.predict(X_test_selected)

# Evaluate the best model's performance
ab_best_accuracy = accuracy_score(y_test, ab_best_model_predictions)
ab_best_precision = precision_score(y_test, ab_best_model_predictions)
ab_best_recall = recall_score(y_test, ab_best_model_predictions)
ab_best_f1 = f1_score(y_test, ab_best_model_predictions)
ab_best_auc = roc_auc_score(y_test, ab_best_model_predictions)

# Store the metrics in a dictionary
ab_best_metrics = {
    'Accuracy': ab_best_accuracy,
    'Precision': ab_best_precision,
    'Recall': ab_best_recall,
    'F1': ab_best_f1,
    'AUC': ab_best_auc
}

# Convert the dictionary to a DataFrame
ab_best_metrics_df = pd.DataFrame([ab_best_metrics])

# Calculate the mean of each row (metric)
ab_best_metrics_df['Mean'] = ab_best_metrics_df.mean(axis=1)

print(ab_best_metrics_df)
#%%
# all models perform sub-optimally compared to ET without parameters tuned. Select the later.