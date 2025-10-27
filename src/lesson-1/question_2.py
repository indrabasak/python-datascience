"""
A list:

L2 = [879, 394, 235, 580, 628, 81, 206, 238, 927, 853, 622, 603, 110, 143, 824,
   324, 343, 506, 634, 325, 258, 900, 960, 286, 449, 890, 921, 170, 888, 851]

Please use Python to answer the following questions (Do not use built-in sum and mean functions):
1. Use a while loop to calculate the sum of the even numbers in L2. (10 points)
2. Write a function to calculate the mean of a list. Use this function to calculate the mean of L2 (10 points)
3.Calculate the sum for elements in L2 which is larger than 500. (10 points)
"""

L2 = [879, 394, 235, 580, 628, 81, 206, 238, 927, 853, 622, 603, 110, 143, 824, 324, 343, 506, 634, 325, 258, 900, 960,
      286, 449, 890, 921, 170, 888, 851]


# 1. Use a while loop to calculate the sum of the even numbers in L2.
index = 0
even_sum = 0
while index < len(L2):
    if L2[index] % 2 == 0:
        even_sum += L2[index]
    index += 1

print("Sum of even numbers in L2:", even_sum)

# 2. Write a function to calculate the mean of a list. Use this function to calculate the mean of L2
def calculate_mean(input_list):
    total = 0
    count = 0
    for num in input_list:
        total += num
        count += 1

    if count > 0:
        mean_value = total / count
    else:
        mean_value = 0

    return mean_value

mean_L2 = calculate_mean(L2)
print("Mean of L2:", mean_L2)

# 3.Calculate the sum for elements in L2 which is larger than 500.
sum_greater_500 = 0
for num in L2:
    if num > 500:
        sum_greater_500 += num

print("Sum of elements in L2 larger than 500:", sum_greater_500)