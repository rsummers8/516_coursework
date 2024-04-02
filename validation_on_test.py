# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:54:55 2024

@author: russe
"""

# evaluate models on test data

#%%
# for non-treated data i am proceeding with et
# validate et on test set 

from sklearn.metrics import roc_auc_score

# Trained ET model
trained_et_model = models[8][1]  # ET is the ninth model in the list of models

# Make predictions on the test set
et_predictions = trained_et_model.predict(X_test_selected)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, et_predictions)
precision = precision_score(y_test, et_predictions)
recall = recall_score(y_test, et_predictions)
f1 = f1_score(y_test, et_predictions)
auc = roc_auc_score(y_test, et_predictions)

# Print the evaluation metrics
print("Evaluation Metrics for Trained ET Model:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)


#%%
# treated data

# Trained ET model on treated data
treated_trained_et_model = models_treated[8][1]  # 9th pos 

# Make predictions on the treated test set
et_predictions_treated = treated_trained_et_model.predict(X_test_treated_selected)

# Calculate evaluation metrics
accuracy_treated = accuracy_score(y_test_treated, et_predictions_treated)
precision_treated = precision_score(y_test_treated, et_predictions_treated)
recall_treated = recall_score(y_test_treated, et_predictions_treated)
f1_treated = f1_score(y_test_treated, et_predictions_treated)
auc_treated = roc_auc_score(y_test_treated, et_predictions_treated)

# Print the evaluation metrics
print("Evaluation Metrics for Trained ET Model on Treated Test Data:")
print("Accuracy:", accuracy_treated)
print("Precision:", precision_treated)
print("Recall:", recall_treated)
print("F1 Score:", f1_treated)
print("AUC:", auc_treated)                        # et model is better on non-treated data suggesting that treating caused more over-fitting 

#%%
# evaluate more non treated data models on test data
# Get the top performing models from the sorted DataFrame
top_models = model_metrics_df_sorted.head(3)  # select n 

# Initialize an empty list to store the evaluation metrics
test_set_metrics = []

# Loop through the top performing models
for index, row in top_models.iterrows():
    model_name = row['Model']
    model = None
    
    # Find the corresponding model instance from the models list
    for model_tuple in models:
        if model_tuple[0] == model_name:
            model = model_tuple[1]
            break
    
    if model:
        # Make predictions on the test set
        predictions = model.predict(X_test_selected)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, predictions)

        # Append the metrics to the list
        test_set_metrics.append({'Model': model_name,
                                 'Accuracy': accuracy,
                                 'Precision': precision,
                                 'Recall': recall,
                                 'F1 Score': f1,
                                 'AUC': auc})

# Convert the list of dictionaries to a DataFrame
test_set_metrics_df = pd.DataFrame(test_set_metrics)

# Display the DataFrame
print(test_set_metrics_df)

#%%
# Calculate averages
test_set_metrics_df['Mean'] = test_set_metrics_df.iloc[:, 1:].mean(axis=1)

# Display the DataFrame with the new column
print(test_set_metrics_df)
#%%
# do the same for treated data
top_models_treated = model_metrics_treated_df_sorted.head(3)  # select n 

# Initialize an empty list to store the evaluation metrics for treated data
test_set_metrics_treated = []

# Loop through the top performing models for treated data
for index, row in top_models_treated.iterrows():
    model_name = row['Model']
    model = None
    
    # Find the corresponding model instance from the models_treated list
    for model_tuple in models_treated:
        if model_tuple[0] == model_name:
            model = model_tuple[1]
            break
    
    if model:
        # Make predictions on the treated test set
        predictions_treated = model.predict(X_test_treated_selected)

        # Calculate evaluation metrics for treated data
        accuracy_treated = accuracy_score(y_test_treated, predictions_treated)
        precision_treated = precision_score(y_test_treated, predictions_treated)
        recall_treated = recall_score(y_test_treated, predictions_treated)
        f1_treated = f1_score(y_test_treated, predictions_treated)
        auc_treated = roc_auc_score(y_test_treated, predictions_treated)

        # Append the metrics to the list for treated data
        test_set_metrics_treated.append({'Model': model_name,
                                         'Accuracy': accuracy_treated,
                                         'Precision': precision_treated,
                                         'Recall': recall_treated,
                                         'F1 Score': f1_treated,
                                         'AUC': auc_treated})

# Convert the list of dictionaries to a DataFrame for treated data
test_set_metrics_df_treated = pd.DataFrame(test_set_metrics_treated)

# Calculate averages for treated data
test_set_metrics_df_treated['Mean'] = test_set_metrics_df_treated.iloc[:, 1:].mean(axis=1)

# Display the DataFrame with the new column for treated data
print(test_set_metrics_df_treated)
#%%
############### non treated data is superior, less overfitting to training data