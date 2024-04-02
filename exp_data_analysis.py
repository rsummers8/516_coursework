# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:20:43 2024

@author: russe
"""
#%%
# exploratory data analysis 

b_cancer.describe()

# refine to present 

stats_canc = b_cancer.describe().T
stats_canc.head(10)
#%%
#%%
# stats for grade 1

grade_one = b_cancer[b_cancer['Grade']==0].describe().T
grade_one.head(10)
#%%
#%%
# stats for grade 2 

grade_two = b_cancer[b_cancer['Grade']==1].describe().T
grade_two.head(10)
#%%
#%%
#compare fold change 
# create dataframe
f_means = pd.DataFrame({'Grade 1':grade_one[:-1]['mean'], 'Grade 2':grade_two[:-1]['mean']})

# calculate fold changes and add to dataframe
fc = (f_means['Grade 1'] - f_means['Grade 2']) / f_means['Grade 2']
f_means['fc'] = fc

# calculate Log2 of fold change and add to data frame
log2_fc = np.log2(np.abs(fc))
f_means['log2_fc'] = log2_fc
f_means = f_means.reset_index().rename(columns = {'index':'feature'})

# Drop the 'Grade' row before sorting
f_means = f_means[f_means['feature'] != 'Grade']

# sort dataframe according to fold change
f_means['abs'] = abs(f_means['fc'])

# Sort by fold absolute change
f_means = f_means.sort_values(by = ['abs'],ascending = False).drop(columns = ['abs'])
f_means.head(10)
#%%
#%%

# plot fold changes 

plt.figure(figsize=(18, 6))
plt.bar(f_means['feature'], f_means['log2_fc'], color='blue', width=0.4)
plt.xticks(f_means['feature'], rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Log2 Fold change')  # Updated y-axis label
plt.tight_layout()
#%%
#%%
# Extract top ten variables from f_means
top_variables = f_means.head(10)['feature']

# Select columns corresponding to top variables from b_cancer
top_variables_data = b_cancer[['Grade'] + top_variables.tolist()]

# Set seaborn plotting aesthetics as default
sns.set()

# Define plotting region (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Create box plot in each subplot
sns.boxplot(x='Grade', y=top_variables_data[top_variables.tolist()[0]], data=top_variables_data, ax=axes[0, 0])
sns.boxplot(x='Grade', y=top_variables_data[top_variables.tolist()[1]], data=top_variables_data, ax=axes[0, 1])
sns.boxplot(x='Grade', y=top_variables_data[top_variables.tolist()[2]], data=top_variables_data, ax=axes[1, 0])
sns.boxplot(x='Grade', y=top_variables_data[top_variables.tolist()[3]], data=top_variables_data, ax=axes[1, 1])

# Amend x-axis tick labels
for ax in axes.flatten():
    ax.set_xticklabels(['Grade 1', 'Grade 2'])

# Set labels and title
for ax, feature in zip(axes.flatten(), top_variables):
    ax.set_xlabel('Grade')
    ax.set_ylabel(feature)

# Add a single title above all subplots
fig.suptitle('Box Plots of Top log2 FC Variables against Grade', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.show()
#%%
#%%
# Extract top ten variables from f_means
top_variables = f_means.head(10)['feature']

# Select columns corresponding to top variables from b_cancer
top_variables_data = b_cancer[['Grade'] + top_variables.tolist()]

# Set seaborn plotting aesthetics as default
sns.set()

# Define plotting region (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Create violin plot in each subplot
sns.violinplot(x='Grade', y=top_variables_data[top_variables.tolist()[0]], data=top_variables_data, ax=axes[0, 0])
sns.violinplot(x='Grade', y=top_variables_data[top_variables.tolist()[1]], data=top_variables_data, ax=axes[0, 1])
sns.violinplot(x='Grade', y=top_variables_data[top_variables.tolist()[2]], data=top_variables_data, ax=axes[1, 0])
sns.violinplot(x='Grade', y=top_variables_data[top_variables.tolist()[3]], data=top_variables_data, ax=axes[1, 1])

# Amend x-axis tick labels
for ax in axes.flatten():
    ax.set_xticklabels(['Grade 1', 'Grade 2'])

# Set labels and title
for ax, feature in zip(axes.flatten(), top_variables):
    ax.set_xlabel('Grade')
    ax.set_ylabel(feature)

# Add a single title above all subplots
fig.suptitle('Violin Plots of Top log2 FC Variables against Grade', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.show()
#%%
#%%
# Extract top ten variables from f_means
top_variables = f_means.head(10)['feature']

# Select columns corresponding to top variables from b_cancer
top_variables_data = b_cancer[['Grade'] + top_variables.tolist()]

# Set seaborn plotting aesthetics as default
sns.set()

# Define plotting region (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Create swarm plot in each subplot
sns.swarmplot(x='Grade', y=top_variables_data[top_variables.tolist()[0]], data=top_variables_data, ax=axes[0, 0], hue='Grade', palette='Set1')
sns.swarmplot(x='Grade', y=top_variables_data[top_variables.tolist()[1]], data=top_variables_data, ax=axes[0, 1], hue='Grade', palette='Set1')
sns.swarmplot(x='Grade', y=top_variables_data[top_variables.tolist()[2]], data=top_variables_data, ax=axes[1, 0], hue='Grade', palette='Set1')
sns.swarmplot(x='Grade', y=top_variables_data[top_variables.tolist()[3]], data=top_variables_data, ax=axes[1, 1], hue='Grade', palette='Set1')

# Amend x-axis tick labels
for ax in axes.flatten():
    ax.set_xticklabels(['Grade 1', 'Grade 2'])
    # remove legend because i've updated x axis labels
    ax.get_legend().remove()

# Set labels and title
for ax, feature in zip(axes.flatten(), top_variables):
    ax.set_xlabel('Grade')
    ax.set_ylabel(feature)

# Add a single title above all subplots
fig.suptitle('Swarm Plots of Top log2 FC Variables against Grade', fontsize=16)

# Adjust legend location
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

# Adjust layout
plt.tight_layout()
plt.show()

#%%
#%%

# see if features are normally dist or not
from scipy import stats

# list features 
features = [col for col in b_cancer.columns if col not in['Grade', 'Subjects']]

# Loop through each feature
for feature in features:
    # Extract the data for the current feature
    data = b_cancer[feature]
    
    # Perform normality test
    stat, p = stats.normaltest(data)
    
    # Set significance level
    alpha = 0.05
    
    # Print the result
    print(f'Feature: {feature}')
    if p > alpha:
        print('  Data is normally distributed')
    else:
        print('  Data is not normally distributed')

#%%
#%%
######## t-test or mann-whitney tests for comparison 
# List features to perform tests on  
features = [col for col in b_cancer.columns if col not in ['Grade', 'Subjects']]

# Initialize lists to store normally and not normally distributed features
normally_distributed_features = []
not_normally_distributed_features = []
results = []

# Loop through each feature
for feature in features:
    # Extract the data for the current feature
    data = b_cancer[feature]
    
    # Perform normality test
    stat, p = stats.normaltest(data)
    
    # Set significance level
    alpha = 0.05
    
    # Determine whether the data is normally distributed and perform the corresponding test
    if p > alpha:
        normally_distributed_features.append(feature)
        grade = 1  # Choose either 0 or 1, as per your preference
        group1 = b_cancer[b_cancer['Grade'] == grade][feature]
        group2 = b_cancer[b_cancer['Grade'] != grade][feature]
        t_stat, p_value = stats.ttest_ind(group1, group2)
        results.append({'Feature': feature, 'Test': 'T-test', 'Statistic': t_stat, 'P-value': p_value})
    else:
        not_normally_distributed_features.append(feature)
        grade = 1  # Choose either 0 or 1, as per your preference
        group1 = b_cancer[b_cancer['Grade'] == grade][feature]
        group2 = b_cancer[b_cancer['Grade'] != grade][feature]
        u_stat, p_value = stats.mannwhitneyu(group1, group2)
        results.append({'Feature': feature, 'Test': 'Mann-Whitney U test', 'Statistic': u_stat, 'P-value': p_value})

# Convert results list into a DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)

#%%
# calculate ROC with AUC and append it to results_df
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Iterate over each continuous predictor variable
for feature in features:
    # Perform logistic regression with the predictor variable
    X = b_cancer[[feature]]
    y = b_cancer['Grade']
    model = LogisticRegression()
    model.fit(X, y)
    
    # Calculate predicted probabilities
    predicted_probabilities = model.predict_proba(X)[:, 1]
    
    # Calculate the AUC
    auc = roc_auc_score(y, predicted_probabilities)
    
    # Find the corresponding row and append the AUC value
    index = results_df[results_df['Feature'] == feature].index
    if not index.empty:
        results_df.loc[index, 'AUC'] = auc

print(results_df)
#%%
# all of these correlation graphs need to have their axis/ keys corrected to be viable for inclusion in report
# create correlation heat map on unscaled data of continuous variables with grade using point-biserial correlation
from scipy.stats import pointbiserialr

# Extract continuous variables excluding 'Grade' and 'Subjects'
continuous_vars = b_cancer.drop(columns=['Grade', 'Subjects'])

# Calculate point-biserial correlation between binary 'Grade' and continuous variables
point_biserial_correlation = {}
for col in continuous_vars.columns:
    corr_coef, p_value = pointbiserialr(b_cancer[col], b_cancer['Grade'])
    point_biserial_correlation[col] = corr_coef

# Convert the dictionary to a DataFrame for visualization
point_biserial_df = pd.DataFrame.from_dict(point_biserial_correlation, orient='index', columns=['Point-Biserial Correlation'])

# transpose df to make horizontal graph
point_biserial_df = point_biserial_df.transpose()

# plot heatmap
plt.figure(figsize=(14, 10))  # Increase the figure size
heatmap = sns.heatmap(point_biserial_df.transpose(), cmap='coolwarm', annot=True, fmt=".2f")
plt.xticks(fontsize=10)  # Rotate x-axis labels, adjust alignment, and font size
plt.yticks(rotation=0, ha='right', fontsize=10)  # Adjust y-axis font size
plt.title('Point-Biserial Correlation Heatmap of Continuous Variables with Grade')
plt.show()


#%%
#%%
# make x axis more clear
# Correlation matrix of just the predictor variables to look for confounding variables 
unscaled_correlation_matrix = continuous_vars.corr()

# Create a heatmap with adjusted x-axis and y-axis tick labels
plt.figure(figsize=(20, 12))
heatmap = sns.heatmap(unscaled_correlation_matrix, annot=False, cmap='coolwarm')
plt.xticks(rotation=90, ha='right', fontsize=8)  # Rotate x-axis labels by 90 degrees, adjust alignment, and font size
plt.yticks(rotation=0, ha='right', fontsize=8)   # Adjust y-axis font size and alignment
plt.title('Correlation Heatmap of Continuous Variables')
plt.show()

#%%     # dimensionality reduction based on colinearity 

# Set colinearity threshold    
correlation_threshold = 0.80        

# Extract just the data without 'Subjects' and 'Grade'
data = b_cancer.drop(columns=['Subjects', 'Grade'])

# Calculate correlation matrix
corr_matrix = data.corr()

# Extract the upper triangle of the correlation matrix -- inter-correlations or colinearity
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Determine features that have a colinearity above threshold
# Need to use the absolute value -- to determine colinearity
to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

# Drop the identified columns from b_cancer
b_cancer_treated = b_cancer.drop(columns=to_drop)   

#%%