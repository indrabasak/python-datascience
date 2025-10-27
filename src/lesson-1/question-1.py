"""
Question 1
Please use the following codes to create a list L1:
"""
import numpy as np

L1 = []

np.random.seed(56)

for i in np.random.randint(0, 100, 10):
    # L1.extend([i] * np.random.randint(0, 100, 1)[0])
    L1.extend([int(i)] * int(np.random.randint(0, 100, 1)[0]))

# print(L1)
np.random.shuffle(L1)
# print(L1)

# 1. What are the unique values?
unique_values = np.unique(L1)
print(unique_values)

# 2. How many unique values?
num_unique_values = len(unique_values)
print(num_unique_values)

# 3. Create a dictionary with the unique items in L1 as dictionary keys and their count as the dictionary values.
value_counts = {int (val): int (L1.count(val)) for val in unique_values}
print(value_counts)

# 4. Which value appears most frequently? The manual comparison is not acceptable.
most_frequent_value = max(value_counts, key=value_counts.get)
print(most_frequent_value)