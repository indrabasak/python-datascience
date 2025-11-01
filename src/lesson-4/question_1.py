"""
Question 1
Upload Assignment4_data.csv Download Assignment4_data.csvinto Python.

Please perform the following steps:

1. Explore the datasets. (10 points)

2. Find and handle missing values in the data.
(It is your choice how you handle the missing data.) ( 20 points)

3. Explore the variable column and convert the "variable” column to dummy
variables and join the dummies to the data. (20 points)

4. Convert the "one” column into 3 bins. (20 points)
"""
import pandas as pd

# Load the dataset
data = pd.read_csv("./data/lesson-4/Assignment4_data.csv")

# 1. Explore the data
print("First 5 rows of the dataset:")
print(data.head())
print("\nStatistical summary of the dataset:")
print(data.describe())
print("\nCount of each column:")
print(data.count())

# 2. Find and handle missing values in the data.
print("\nMissing values in each column:")
print(data.isnull().sum())
# Handling missing values by filling them with the mean of the column
# Handling missing values: fill numeric columns with mean, non-numeric with mode
for col in data.columns:
    if data[col].dtype == 'O':  # object/non-numeric
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

print(data)

# 3. Explore the variable column and convert the "variable” column to
# dummy variables and join the dummies to the data.
print("\nUnique values in 'variable' column:")
print(data['variable'].unique())

"""
pd.get_dummies converts categorical variable(s) into dummy/indicator variables 
(one-hot encoding). It creates new columns for each unique value in the 
specified column, assigning 1 or 0 to indicate the presence of each category. 
This is useful for preparing data for machine learning models that require numeric input.
"""
dummies = pd.get_dummies(data['variable'], prefix='var', drop_first=True)
print("\nData with dummy variables added:")
print(dummies.head())

print("\nJoining dummy variables to the original data:")
data = pd.concat([data, dummies], axis=1)
print(data.head())

# 4. Convert the "one” column into 3 bins.
print("\nBinning the 'one' column into 3 bins:")
data['one_binned'] = pd.cut(data['one'], bins=3, labels=['Low', 'Medium', 'High'])
print(data[['one', 'one_binned']].head())
