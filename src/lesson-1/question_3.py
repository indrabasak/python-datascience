"""
Question 3
1. Implement the function pow(x, n), which calculates x raised to the power n (xn).
Please don't use x**n. (20pts)

2. Calculate pow(2, 10) and pow(3, -3). (10 pts)
*It is recommended to try to use Jupyter Notebook to finish assignments since it is a convenient tool for assignments and reports. You can find the video of Jupyter Notebook tutorial in the Media Gallery.

*Submit your Python file with your results to this assignment with the extension (.py) if you are using Spyder or your Jupyter Notebook with the extension (.ipynb). In both cases, make sure to upload the printout of your code file as a PDF file so I can add my comments.
With that being said, you have to upload two files per assignment submission!
"""

# 1. Implement the function pow(x, n), which calculates x raised to the power n (xn).
def pow(x, n):
    if n == 0:
        return 1
    elif n > 0:
        result = 1
        for _ in range(n):
            result *= x
        return result
    else:  # n < 0
        result = 1
        for _ in range(-n):
            result *= x
        return 1 / result

# 2. Calculate pow(2, 10) and pow(3, -3).
result_2_10 = pow(2, 10)
print(result_2_10)
result_3_neg3 = pow(3, -3)
print(result_3_neg3)