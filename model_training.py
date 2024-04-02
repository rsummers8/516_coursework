# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:30:45 2024

@author: russe
"""

# compare models on training data
#%%
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
#%%
# firstly for non-treated data   # this section creates a boxplot based on accuracy scores of training data
# prepare configuration for cross validation test harness


# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVM', SVC(kernel='poly')))
models.append(('RF', RandomForestClassifier()))
models.append(('BG', BaggingClassifier()))
models.append(('ET', ExtraTreesClassifier()))
models.append(('SGDC', SGDClassifier()))
models.append(('NN', Perceptron()))
models.append(('XGB', XGBClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('AB', AdaBoostClassifier()))
models.append(('MLP', MLPClassifier()))
models.append(('LGBM', LGBMClassifier()))
models.append(('CatBoost', CatBoostClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

# sort the data and labels
X = X_train_selected
y = y_train
Y = y

# Accuracy
print('Calculating accuracies')
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.05)
plt.tight_layout()
plt.show()

#%%
# prepare configuration for cross validation test harness
seed = 6
np.random.seed(seed)
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVM', SVC(kernel='poly')))
models.append(('RF', RandomForestClassifier()))
models.append(('BG', BaggingClassifier()))
models.append(('ET', ExtraTreesClassifier()))
models.append(('SGDC', SGDClassifier()))
models.append(('NN', Perceptron()))
models.append(('XGB', XGBClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('AB', AdaBoostClassifier()))
models.append(('MLP', MLPClassifier()))
models.append(('LGBM', LGBMClassifier()))
models.append(('CatBoost', CatBoostClassifier()))

# evaluate each model in turn
# evaluate each model in turn
results = []
names = []
metrics = ['accuracy', 'precision', 'recall', 'f1']
scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}

# sort the data and labels
X = X_train_selected
y = y_train
# Initialize an empty list to store the data
data = []

print('Calculating metrics')

# Loop through each model
for name, model in models:
    # Fit the model on the training data
    model.fit(X_train_selected, y_train)
    
    # Make predictions on the training data
    predictions = model.predict(X_train_selected)
    
    # Calculate evaluation metrics
    accuracy_mean = accuracy_score(y_train, predictions)
    precision_mean = precision_score(y_train, predictions)
    recall_mean = recall_score(y_train, predictions)
    f1_mean = f1_score(y_train, predictions)
    
    # Append the metrics to the data list
    data.append({'Model': name, 'Accuracy_mean': accuracy_mean,
                 'Precision_mean': precision_mean,
                 'Recall_mean': recall_mean,
                 'F1_mean': f1_mean})

# Create DataFrame from the collected data
model_metrics_df = pd.DataFrame(data)

# Display the DataFrame
print(model_metrics_df)


#%%   # append the averages of all the mean performance metriccs by model  
# Calculate the average of each numerical value in a row and add a column with those averages for each model
model_metrics_df['Mean'] = model_metrics_df.iloc[:, 1:].mean(axis=1)

# Display the DataFrame with the new column
print(model_metrics_df)

#%%   # plot in descending order
# Sort the DataFrame by the 'Average' column in descending order
model_metrics_df_sorted = model_metrics_df.sort_values(by='Mean', ascending=False)
#%%
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(model_metrics_df_sorted['Model'], model_metrics_df_sorted['Mean'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Average of Performance Metrics')
plt.title('Average Performance by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
#%%  ######################################################################***
# create different list of trained models for treated data
seed = 6
np.random.seed(seed)
# prepare models
models_treated = []
models_treated.append(('LR', LogisticRegression()))
models_treated.append(('LDA', LinearDiscriminantAnalysis()))
models_treated.append(('KNN', KNeighborsClassifier()))
models_treated.append(('DT', DecisionTreeClassifier()))
models_treated.append(('GNB', GaussianNB()))
models_treated.append(('SVM', SVC(kernel='poly')))
models_treated.append(('RF', RandomForestClassifier()))
models_treated.append(('BG', BaggingClassifier()))
models_treated.append(('ET', ExtraTreesClassifier()))
models_treated.append(('SGDC', SGDClassifier()))
models_treated.append(('NN', Perceptron()))
models_treated.append(('XGB', XGBClassifier()))
models_treated.append(('GB', GradientBoostingClassifier()))
models_treated.append(('AB', AdaBoostClassifier()))
models_treated.append(('MLP', MLPClassifier()))
models_treated.append(('LGBM', LGBMClassifier()))
models_treated.append(('CatBoost', CatBoostClassifier()))

# evaluate each model in turn
# evaluate each model in turn
results_t = []
names_t = []
metrics_t = ['accuracy', 'precision', 'recall', 'f1']
scoring_t = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}

# sort the data and labels
X_t = X_train_treated_selected
y_t = y_train_treated

# Initialize an empty list to store the data
data_treated = []

print('Calculating metrics for treated data')

# Loop through each model
for name, model in models_treated:
    # Fit the model on the treated training data
    model.fit(X_train_treated_selected, y_train_treated)
    
    # Make predictions on the treated training data
    predictions_treated = model.predict(X_train_treated_selected)
    
    # Calculate evaluation metrics
    accuracy_mean_treated = accuracy_score(y_train_treated, predictions_treated)
    precision_mean_treated = precision_score(y_train_treated, predictions_treated)
    recall_mean_treated = recall_score(y_train_treated, predictions_treated)
    f1_mean_treated = f1_score(y_train_treated, predictions_treated)
    
    # Append the metrics to the data list
    data_treated.append({'Model': name, 'Accuracy_mean': accuracy_mean_treated,
                 'Precision_mean': precision_mean_treated,
                 'Recall_mean': recall_mean_treated,
                 'F1_mean': f1_mean_treated})

# Create DataFrame from the collected data
model_metrics_treated_df = pd.DataFrame(data_treated)

# Display the DataFrame
print(model_metrics_treated_df)
#%% # same for treated   
# append the averages of all the mean performance metriccs by model  
# Calculate the average of each numerical value in a row and add a column with those averages for each model
model_metrics_treated_df['Mean'] = model_metrics_treated_df.iloc[:, 1:].mean(axis=1)

# Display the DataFrame with the new column
print(model_metrics_treated_df)
#%%
# plot in descending order
# Sort the DataFrame by the 'Average' column in descending order
model_metrics_treated_df_sorted = model_metrics_treated_df.sort_values(by='Mean', ascending=False)
#%%
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(model_metrics_treated_df_sorted['Model'], model_metrics_treated_df_sorted['Mean'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Average of Performance Metrics')
plt.title('Average Performance by Model for Treated Data')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%%
# this cell creates a stacked chart of the different performance metrics by model for the training data
# prepare configuration for cross validation test harnes
# Loop through each model in models_treated
print('Calculating metrics')
for name_t, model in models_treated:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = {metric: model_selection.cross_val_score(model, X_t, y_t, cv=kfold, scoring=scoring_t[metric]) for metric in metrics_t}
    results_t.append(cv_results)
    names_t.append(name_t)
    print(f"{name_t}:")
    for metric, scores in cv_results.items():
        print(f"  {metric}: {scores.mean()} ({scores.std()})")

# Prepare data for plotting
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_colors = ['b', 'g', 'r', 'orange']
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting
for i, metric in enumerate(metrics_t):
    scores = [result[metric] for result in results_t]
    ax.bar(names_t, [np.mean(score) for score in scores], color=metric_colors[i], label=metric_names[i], alpha=0.7, bottom=np.sum([np.mean(score) for score in scores[:i]], axis=0))

ax.set_ylabel('Score')
ax.set_title('Model Performance Metrics')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%%  # this cell creates a data frame of the mean performance metics by model for treated data
# Initialize an empty list to store the data
seed = 6
data_treated = []

print('Calculating metrics')
for name_t, model in models_treated:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = {metric: model_selection.cross_val_score(model, X_t, y_t, cv=kfold, scoring=scoring_t[metric]) for metric in metrics_t}
    accuracy_mean = cv_results['accuracy'].mean()
    precision_mean = cv_results['precision'].mean()
    recall_mean = cv_results['recall'].mean()
    f1_mean = cv_results['f1'].mean()
    data_treated.append({'Model': name_t, 'Accuracy_mean': accuracy_mean,
                 'Precision_mean': precision_mean,
                 'Recall_mean': recall_mean,
                 'F1_mean': f1_mean})

# Create DataFrame from the collected data
model_metrics_treated_df = pd.DataFrame(data_treated)

# Display the DataFrame
print(model_metrics_treated_df)
#%%