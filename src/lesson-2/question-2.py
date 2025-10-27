"""
Perform the following actions:

Use x = np.random.randint(0, 1000, size = (10, 10)) to generate 10x10
array and use a for loop to find out how many even numbers are in it. (25 points)

Randomly generate an 8x9 array from a normal distribution with mean = 1,
sigma = 0.5. Calculate the mean of elements whose indexes have a
relation of (i+j)%5 == 0  (i is row index and j is column index).
"""

import numpy as np

np.random.seed(100)

# Use x = np.random.randint(0, 1000, size = (10, 10)) to generate 10x10
# array and use a for loop to find out how many even numbers are in it. (25 points)
array_10x10 = np.random.randint(0,    1000, size=(10, 10))
print("Array 10x10:")
print(array_10x10)

# 1. Find out how many even numbers are in the array
even_count = 0
for i in range(10):
    for j in range(10):
        if array_10x10[i, j] % 2 == 0:
            even_count += 1

print("Number of even numbers in the array:", even_count)

# 2. Randomly generate an 8x9 array from a normal distribution with mean = 1,
# sigma = 0.5. Calculate the mean of elements whose indexes have a
# relation of (i+j)%5 == 0  (i is row index and j is column index).
mean = 1
sigma = 0.5
array_8x9 = np.random.normal(mean, sigma, size=(8, 9))
print("Array 8x9 from normal distribution:")
print(array_8x9)

# Calculate the mean of elements whose indexes have a relation of (i+j)%5 == 0
selected_elements = []
for i in range(8):
    for j in range(9):
        if (i + j) % 5 == 0:
            selected_elements.append(array_8x9[i, j])

mean_selected  = np.mean(selected_elements)
print("Mean of selected elements:", mean_selected)
