"""
Please use mtcars dataset Download mtcars data set to perform the following actions:
1. Plot am-based histogram to compare mpg (20 points)
2. Use scatterplot to plot mpg VS. hp (20 points)
3. Create a scatterplot matrix for new data consisting of columns [disp, hp, drat, wt, qsect]. (20 points)
4. Create boxplots for new data consisting of columns [disp, hp, drat, wt, qsect]. (20 points)
5. Use plots to answer which variable has the most impact on mpg. (20 points)
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the mtcars dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv"
mtcars = pd.read_csv("./data/lesson-6/mtcars.csv")
mtcars.columns = mtcars.columns.str.strip()  # Clean column names
# display data
print("Mtcars Dataset:")
print(mtcars.head())

# In the mtcars dataset, the am column represents the transmission type of the car:
# 0: Automatic transmission
# 1: Manual transmission

# 1. Plot am-based histogram to compare mpg
plt.figure(figsize=(10, 6))

# In a histogram plot, bins specifies the number of intervals (bars) into which the
# data is divided. Each bin groups data points that fall within its range, helping
# visualize the distribution of values.
# Example:
# bins=10 creates 10 bars, each representing a range of mpg values.
# More bins = finer detail; fewer bins = more general overview.
mtcars[mtcars['am'] == 0]['mpg'].plot(kind='hist', alpha=0.5, label='Automatic', bins=10)
mtcars[mtcars['am'] == 1]['mpg'].plot(kind='hist', alpha=0.5, label='Manual', bins=10)
plt.title('MPG Distribution by AM')
plt.xlabel('Miles Per Gallon (mpg)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 2. Use scatterplot to plot mpg VS. hp
plt.figure(figsize=(10, 6))
plt.scatter(mtcars['hp'], mtcars['mpg'], alpha=0.7)
plt.title('MPG vs HP')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles Per Gallon (mpg)')
plt.show()

# 3. Create a scatterplot matrix for new data consisting of columns [disp, hp, drat, wt, qsect].
selected_cols = ['disp', 'hp', 'drat', 'wt', 'qsec']
fig, axes = plt.subplots(1, len(selected_cols), figsize=(20, 5))
for i, col in enumerate(selected_cols):
    axes[i].scatter(mtcars[col], mtcars['mpg'], alpha=0.7)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('mpg')
    axes[i].set_title(f'mpg vs {col}')
plt.tight_layout()
plt.show()

# 4. Create boxplots for new data consisting of columns [disp, hp, drat, wt, qsect].
plt.figure(figsize=(12, 8))
bplot = mtcars[selected_cols].boxplot(return_type='dict', patch_artist=True)
plt.title('Boxplots for disp, hp, drat, wt, qsec')
plt.ylabel('Values')

colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightgrey']
# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

# 5. Use plots to answer which variable has the most impact on mpg.
correlations = mtcars[selected_cols + ['mpg']].corr()['mpg'].drop('mpg')
print("Correlation of selected variables with mpg:")
print(correlations)
correlations.plot(kind='bar', color='skyblue', figsize=(8, 5))
plt.title('Correlation of Variables with MPG')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Variable')
plt.show()

