"""
Data Set Information
This dataset is composed of a range of biomedical voice measurements from 42 people with
early-stage Parkinson's disease recruited to a six-month trial of a telemonitoring device for
remote symptom progression monitoring. The recordings were automatically captured in the patient's homes.

Columns in the table contain subject number, subject age, subject gender, time interval from
baseline recruitment date, motor UPDRS, total UPDRS, and 16 biomedical voice measures.

Each row corresponds to one of 5,875 voice recordings from these individuals.

The main aim of the data is to predict the motor and total UPDRS scores ('motor_UPDRS' and 'total_UPDRS')
from the 16 voice measures.

Perform exploratory analysis on the data and Remove motor_UPDRS column. (10 points)
1. Use cross-validation to build a linear regression model to predict total_UPDRS. (25 points)
2. Use cross-validation to build a regression tree model to predict total_UPDRS. (25 points)
3. Use cross-validation to build a neural network model to predict total_UPDRS. (25 points)
4. Compare their performance with MAE, which model has better performance? Is there any way to improve the model? (5 points)
5. Try to optimize the tree model or neural network model (Choose one). (10 points)
"""
import asyncio

import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score

parkinsons_updrs = pd.read_csv("./data/lesson-9/parkinsons+telemonitoring/parkinsons_updrs.data")

def explore_data():
    print(parkinsons_updrs.head())
    print(parkinsons_updrs.describe())

def run_linear_regression():
    """
    1. Use cross-validation to build a linear regression model to predict total_UPDRS. (25 points)
    :return:
    """
    # Data Preparation
    # Dropping the motor_UPDRS and total_UPDRS columns because:
    # 1. motor_UPDRS is not needed for predicting total_UPDRS and should be removed as per the instructions.
    # 2. total_UPDRS is the target variable you want to predict, so it should not be included in the feature set (X).
    # This ensures your model only uses the remaining columns as input features.
    X = parkinsons_updrs.drop(columns=['motor_UPDRS', 'total_UPDRS'])

    # Assigns the target variable y for the regression model. It selects the
    # total_UPDRS column from the parkinsons_updrs DataFrame, which is the value you want to
    # predict using your model. This is necessary for supervised learning,
    # where you need both input features (X) and the target output (y).
    y = parkinsons_updrs.total_UPDRS

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    print("Linear Regression Data Preparation Complete")

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
    r2_score_train = metrics.r2_score(y_train, lreg_train_pred)
    print("Train R2 Score:", r2_score_train)
    mse_train = metrics.mean_squared_error(y_train, lreg_train_pred)
    print("Train Mean Squared Error:", mse_train)


    lreg_test_pred = lreg.predict(X_test)
    print("Test Predictions:", lreg_test_pred)
    r2_score_test = metrics.r2_score(y_test, lreg_test_pred)
    print("Test R2 Score:", r2_score_test)
    mse_test = metrics.mean_squared_error(y_test, lreg_test_pred)
    print("Test Mean Squared Error:", mse_test)

    # Cross Validation
    cv_scores = cross_val_score(lreg, X, y, cv=5, scoring='neg_mean_absolute_error')
    print("Cross-Validation MAE Scores:", -cv_scores)
    print("Average Cross-Validation MAE:", -cv_scores.mean())


async def main():
    print("--------------")
    explore_data()
    run_linear_regression()


asyncio.run(main())