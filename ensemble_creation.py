# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:58:31 2024

@author: russe
"""
### non- treated data
## improving model via ensemble method
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import cross_val_predict

#%%

# Train RandomForestClassifier
trained_rf_model = models[6][1]  # RandomForestClassifier is the seventh model in the list

# Train AdaBoostClassifier
trained_ab_model = models[13][1]  # AdaBoostClassifier is the fourteenth model in the list

#%%
from sklearn.ensemble import VotingClassifier
seed = 6 

np.random.seed(seed)
# Define the ensemble models
ensemble_models = [
    ('ET', trained_et_model),  # Assuming trained_et_model is the ExtraTreesClassifier
    ('RF', trained_rf_model),  # Assuming trained_rf_model is the RandomForestClassifier
    ('AB', trained_ab_model)   # Assuming trained_ab_model is the AdaBoostClassifier
]

# Create the Voting Classifier
voting_clf = VotingClassifier(estimators=ensemble_models, voting='hard')  # 'hard' for majority voting

# Fit the Voting Classifier on the training data
voting_clf.fit(X_train_selected, y_train)

# Make predictions on the test set
ensemble_predictions = voting_clf.predict(X_test_selected)

# Evaluate the ensemble performance
e_accuracy = accuracy_score(y_test, ensemble_predictions)
e_precision = precision_score(y_test, ensemble_predictions)
e_recall = recall_score(y_test, ensemble_predictions)
e_f1 = f1_score(y_test, ensemble_predictions)
e_auc = roc_auc_score(y_test, ensemble_predictions)

# Store the metrics in a dictionary
ensemble_metrics = {
    'Accuracy': e_accuracy,
    'Precision': e_precision,
    'Recall': e_recall,
    'F1 Score': e_f1,
    'AUC': e_auc
}

# Convert the dictionary to a DataFrame
ensemble_metrics_df = pd.DataFrame([ensemble_metrics])

# Calculate the average of each numerical value in a row and add a column with those averages
ensemble_metrics_df['Mean'] = ensemble_metrics_df.mean(axis=1)

# Display the DataFrame with the new column
print(ensemble_metrics_df)
# ensemble has same metrics as ET alone
#%%