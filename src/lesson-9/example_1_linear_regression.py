"""
The mechanism of the three regression models introduced in this lesson have been introduced
in the course IE575. Here we only provide a quick walk-through of
how to use regression algorithms with Scikit-learn package in Python.

At the end of this lesson, using scikit-learn, the student will be able to:
1. Build and evaluate Linear Regression models
2. Build and evaluate Decision Tree models
3. Build and evaluate Nerual networks models
"""

import asyncio

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import probplot
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

mtcars = pd.read_csv("./data/lesson-6/mtcars.csv")

def explore_data():
    print(mtcars.describe())

def linear_regression():
    """
    We will build a simple linear regression model with the disp column as X and mpg column as y.
    This simple model allows us to visualize the data and the model. For a complicated model,
    you can assign more columns to X. The expand_dims is used to convert the one-dimensional
    array to a two-dimensional array.
    :return:
    """
    # Data Preparation
    X = np.expand_dims(mtcars.disp, axis=1)
    y = mtcars.mpg

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # Model Generation
    lreg = linear_model.LinearRegression()
    fit = lreg.fit(X_train, y_train)
    print(fit)

    coef = lreg.coef_
    intercept = lreg.intercept_
    print("Coefficients:", coef)
    print("Intercept:", intercept)

    # Model Evaluation
    # The performance of the model can be evaluated by R-square and mean squared error
    lreg_train_pred = lreg.predict(X_train)
    print("Train Predictions:", lreg_train_pred)

    lreg_test_pred = lreg.predict(X_test)
    print("Test Predictions:", lreg_test_pred)

    r2_score_train = metrics.r2_score(y_train, lreg_train_pred)
    print("Train R2 Score:", r2_score_train)

    mse_train = metrics.mean_squared_error(y_train, lreg_train_pred)
    print("Train Mean Squared Error:", mse_train)

    r2_score_test = metrics.r2_score(y_test, lreg_test_pred)
    print("Test R2 Score:", r2_score_test)

    mse_test = metrics.mean_squared_error(y_test, lreg_test_pred)
    print("Test Mean Squared Error:", mse_test)

    # Residual Analysis
    # Plot the original data points and the fitted line - Scatter plot
    # to visualize the linear relationship between disp and mpg
    plt.figure()
    plt.scatter(X, y, color='black', label='True Values')
    plt.plot(X, lreg.predict(X), color='blue', linewidth=3, label='Linear Regression Model')
    plt.xlabel("Displacement")
    plt.ylabel("MPG")
    plt.legend()
    plt.show()
    # From this plot, we can tell that Mpg has a linear relationship with Displacement.

    # Plot the residuals to check for homoscedasticity and independence
    y_pred = lreg.predict(X)
    residual = y - y_pred
    plt.figure()
    plt.scatter(y_pred, residual)
    plt.hlines(0, xmin=10, xmax=30)
    plt.xlim([10, 30])
    plt.xlabel("Predicted Values")
    plt.ylabel("Residual")
    plt.show()
    # It looks like the residual is equally distributed around y=0 line
    # and there is no obvious pattern in the residual.

    # QQ Plot to check for normality of residuals
    # QQ plot shows how well the distribution of residuals fit the normal distribution.
    # In Python, we will use the probplot in Scipy package to realize it:
    plt.figure()
    probplot(residual, plot = plt)
    plt.show()


async def main():
    print("--------------")
    explore_data()
    linear_regression()


asyncio.run(main())
