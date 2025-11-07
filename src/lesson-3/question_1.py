"""
Perform the following actions:
1. Import data mtcars.csv Download mtcars.csvinto Python. (10 points)

2. Explore the data and perform a statistical analysis of the data. (30 points)

3. Analyze mpg for cars with different gear, and show your findings. (20 points)

4. Analyze mpg for cars with different carb, and show your findings. (20 points)

5. Find out which attribute has the most impact on mpg. (20 points)
"""
import numpy as np
import pandas as pd

# 1. Load the mtcars dataset
data = pd.read_csv("./data/lesson-3/mtcars.csv")

# 2. Explore the data
print("First 5 rows of the dataset:")
print(data.head())

print("\nStatistical summary of the dataset:")
print(data.describe())

print("\nnCount of each column:")
print(data.count())

print("\nMinimum values of each column:")
print(data.min())

print("\nMaximum values of each column:")
print(data.max())

print("\nCompute index values at which minimumvalue obtained:")
print(data.idxmin())

print("\nCompute index values at which maximumvalue obtained:")
print(data.idxmax())

# A quantile is a statistical value that divides a dataset into equal-sized, ordered segments.
# ordered segments. For example, the 0.25 quantile (also called the first quartile)
# is the value below which 25% of the data falls. Quantiles help summarize
# the distribution of data and are useful for understanding spread and outliers.
print("\nCompute the 25% quantile values of the dataset:")
# Compute the 25% quantile values for numeric columns only
print(data.select_dtypes(include=[np.number]).quantile(0.25))

print("Sum of mpg column:", data["mpg"].sum())

print("\nMean of numeric columns:")
print(data.select_dtypes(include=[np.number]).mean())

print("\nMedian of numeric columns:")
print(data.select_dtypes(include=[np.number]).median())

print("\nStandard deviation of numeric columns:")
print(data.select_dtypes(include=[np.number]).std())

# 3. Analyze mpg for cars with different gear
print("\nMPG analysis for cars with different gear:")
mpg_by_gear = data.groupby("gear")["mpg"].mean()
print(mpg_by_gear)

# 4. Analyze mpg for cars with different carb
print("\nMPG analysis for cars with different carb:")
mpg_by_carb = data.groupby("carb")["mpg"].mean()
print(mpg_by_carb)

# 5. Find out which attribute has the most impact on mpg
"""
The corr function in pandas computes the pairwise correlation coefficients 
between columns in a DataFrame. It measures how strongly two variables are related, 
with values ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation). 
The result is a correlation matrix showing these relationships for all numeric columns.
"""
correlation_matrix = data.drop(columns=["model"]).corr()
mpg_correlations = correlation_matrix["mpg"].drop("mpg")  # Exclude self-correlation
most_impactful_attribute = mpg_correlations.abs().idxmax()
impact_value = mpg_correlations[most_impactful_attribute]
print("\nAttribute with the most impact on mpg:", most_impactful_attribute)
print("Correlation value:", impact_value)

mpg_corr = data.drop(columns=["model"]).corr()['mpg'].sort_values(ascending=False)
print("\nCorrelation of features with MPG:")
print(mpg_corr)