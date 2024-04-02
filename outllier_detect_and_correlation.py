# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:16:48 2024

@author: russe
"""
#%%
# find outliers
from sklearn.neighbors import LocalOutlierFactor

# List of features to exclude from outlier detection
exclude_features = ['Subjects', 'Grade']  # Specify the features to exclude

# Identify float columns and exclude specific features
float_columns = b_cancer.select_dtypes(include=['float']).columns
float_columns = [col for col in float_columns if col not in exclude_features]

# Subset the DataFrame to include only the selected float columns
float_data = b_cancer[float_columns]
#%%
#%% 
# Instantiate and fit LOF model
lof_model = LocalOutlierFactor(contamination=0.025)  # toggle contam
outlier_scores = lof_model.fit_predict(float_data)

# Identify outliers
outliers = b_cancer.iloc[outlier_scores == -1]

# Compute correlations between each feature and their outlier scores
correlations = float_data.corrwith(pd.Series(outlier_scores))

# Sort correlations by absolute values
sorted_correlations = correlations.abs().sort_values(ascending=False)
#%%
#%%
# Visualize feature-outlier score correlations
plt.figure(figsize=(10, 6))
sorted_correlations.plot(kind='bar')
plt.xlabel('Feature')
plt.ylabel('Absolute Correlation with Outlier Score')
plt.title('Feature-Outlier Score Correlations')
plt.show()

# here's how I found rows that were in the 2.5% of outliers
# but, I computed correlation scores of each feature against their outlier score
# and some features to consider/ reaffrim choices about feature selection as they are correlated highly with outlier scores
# however this could be just guiding as outliers may still be truly representative of data  
#%%
# create df where outliers are exluded for comparison later on 

index_of_outliers = [36,62,77]

b_cancer_treated = b_cancer_treated.drop(index=index_of_outliers)

print(b_cancer_treated.shape)

#%%