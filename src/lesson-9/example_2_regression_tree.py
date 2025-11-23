"""
We have introduced a Classification tree in the previous lesson, here we show how to use
a tree model (DecisionTreeRegressor) for regression tasks.
DecisionTreeClassifier has similar parameters, attribute and methods with DecisionTreeClassifier
"""
import asyncio

import pandas as pd
from graphviz import Source
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics

mtcars = pd.read_csv("./data/lesson-6/mtcars.csv")


def explore_data():
    print(mtcars.describe())


def regression_tree():
    # Data Preparation
    # For CART regression tree model, all variables of the mtcars dataset will be used.
    # First, we split the data into training and test sets
    X = mtcars.loc[:, "cyl":]
    y = mtcars.mpg

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # Model Generation
    tree_model = tree.DecisionTreeRegressor(min_samples_leaf=5, random_state=39)
    fit = tree_model.fit(X_train, y_train)
    print(fit)

    # how you can plot the tree structure by using the graphviz package
    src = Source(tree.export_graphviz(tree_model, out_file=None, feature_names=X.columns, filled=True))
    # src.view()

    # Model Evaluation
    tree_train_pred = tree_model.predict(X_train)
    print("Train Predictions:", tree_train_pred)
    r2_score_train = metrics.r2_score(y_train, tree_train_pred)
    print("Train R2 Score:", r2_score_train)
    mse_train = metrics.mean_squared_error(y_train, tree_train_pred)
    print("Train Mean Squared Error:", mse_train)

    tree_test_pred = tree_model.predict(X_test)
    print("Test Predictions:", tree_test_pred)
    r2_score_test = metrics.r2_score(y_test, tree_test_pred)
    print("Test R2 Score:", r2_score_test)
    mse_test = metrics.mean_squared_error(y_test, tree_test_pred)
    print("Test Mean Squared Error:", mse_test)
    # It looks the performance on training data is much better than the test data, which is a
    # strong indication of overfitting. Overfitting is a common issue

    # Model Improvement:
    # Let's increase the min samples size to see if the overfitting can be solved.
    # The following example shows how we set the min sample size to 8:
    tree_model_1 = tree.DecisionTreeRegressor(min_samples_leaf=8, random_state=39)
    fit_1 = tree_model_1.fit(X_train, y_train)
    print(fit_1)

    Source(tree.export_graphviz(tree_model_1, out_file=None, feature_names=X.columns, filled=True))

    tree_train_pred_1 = tree_model_1.predict(X_train)
    print("Train Predictions:", tree_train_pred_1)
    r2_score_train_1 = metrics.r2_score(y_train, tree_train_pred_1)
    print("Train R2 Score:", r2_score_train_1)
    mse_train_1 = metrics.mean_squared_error(y_train, tree_train_pred_1)
    print("Train Mean Squared Error:", mse_train_1)

    tree_test_pred_1 = tree_model_1.predict(X_test)
    print("Test Predictions:", tree_test_pred_1)
    r2_score_test_1 = metrics.r2_score(y_test, tree_test_pred_1)
    print("Test R2 Score:", r2_score_test_1)
    mse_test_1 = metrics.mean_squared_error(y_test, tree_test_pred_1)
    print("Test Mean Squared Error:", mse_test_1)


async def main():
    print("--------------")
    explore_data()
    regression_tree()


asyncio.run(main())
