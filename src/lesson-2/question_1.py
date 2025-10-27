"""
Question 1
Perform the following actions:

Use the randn function to create an array with a dimension of 5X5 and use a
for loop to calculate the sum of all elements in the diagonal of the array. (25 points)
Choose any three functions to apply to this array. (25 points)
"""

import numpy as np

np.random.seed(42)
array_5x5 = np.random.randn(5, 5)
print("Array 5x5:")
print(array_5x5)

# 1. Calculate the sum of all elements in the diagonal of the array
diagonal_sum = 0
for i in range(5):
    diagonal_sum += array_5x5[i, i]

print(diagonal_sum)

# 2. Choose any three functions to apply to this array
# Function 1: Calculate the mean of the array
mean_value = np.mean(array_5x5)
print("Mean of the array:", mean_value)

# Function 2: Calculate the standard deviation of the array
std_deviation = np.std(array_5x5)
print("Standard Deviation of the array:", std_deviation)

# Function 3: Transpose the array
transposed_array = np.transpose(array_5x5)
print("Transposed Array:")
print(transposed_array)