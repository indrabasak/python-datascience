"""
Class MLPRegressor is the neural network model for regression tasks. It uses a multi-layer
 perception to train the data with the backpropagation method and the
square error as the loss function.
"""
import asyncio

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neural_network, metrics

mtcars = pd.read_csv("./data/lesson-6/mtcars.csv")


def explore_data():
    print(mtcars.describe())


def run_neural_network():
    # Data Preparation
    # For Neural Network, preprocessing data is recommended. First, let’s convert
    # gear and carb variables to dummy variables

    # Remove non-numeric columns (e.g., 'model' or 'car name')
    mtcars_numeric = mtcars.drop(columns=['model'])  # Replace 'model' with the actual column name
    dummies_gear = pd.get_dummies(mtcars_numeric.gear, prefix="gear")
    print(dummies_gear[:3])
    dummies_carb = pd.get_dummies(mtcars_numeric.carb, prefix="carb")
    print(dummies_carb[:3])

    # we need to join dummies_gear and dummies_carb to the data
    # (after remove the original gear and carg variable):
    mtcars_dummies = mtcars_numeric.iloc[:, :10].join(dummies_gear)
    mtcars_dummies = mtcars_dummies.join(dummies_carb)
    print(mtcars_dummies.columns)

    # Then, we need to rescale all variables to the range [0, 1].
    # We use a slightly different approach compared to the lesson 8. First, we fit the
    # scaler with training data instead of using fit_transform function and then transform
    # the train and test dataset
    train, test = train_test_split(mtcars_dummies, test_size=0.3, random_state=123)

    print(" 0 ++++++++++++++++++++++++++++")
    scaler = MinMaxScaler()
    scaler.fit(train)
    print(" 0a ++++++++++++++++++++++++++++")

    train = scaler.transform(train)
    test = scaler.transform(test)



    X_train_scaled = train[:, 1:]
    X_test_scaled = test[:, 1:]
    y_train_scaled = train[:, 0]
    y_test_scaled = test[:, 0]

    print(" 1 ++++++++++++++++++++++++++++")

    # Model Generation:
    # Now the data is ready for model generation. Let’s import the model and create an
    # MLPRegressor object with 50 hidden nodes and using logistic as the
    # activation function
    nn_model = neural_network.MLPRegressor(hidden_layer_sizes=10, activation="logistic", max_iter=10000, random_state=21)
    nn_model.fit(X_train_scaled, y_train_scaled)
    print("Coefficients:", nn_model.coefs_)
    print("Intercepts:", nn_model.intercepts_)

    # Model Evaluation
    nn_train_pred = nn_model.predict(X_train_scaled)
    print("Train Predictions:", nn_train_pred)
    r2_score_train = metrics.r2_score(y_train_scaled, nn_train_pred)
    print("Train R2 Score:", r2_score_train)
    mse_train = metrics.mean_squared_error(y_train_scaled, nn_train_pred)
    print("Train Mean Squared Error:", mse_train)
    nn_test_pred = nn_model.predict(X_test_scaled)
    print("Test Predictions:", nn_test_pred)
    r2_score_test = metrics.r2_score(y_test_scaled, nn_test_pred)
    print("Test R2 Score:", r2_score_test)
    mse_test = metrics.mean_squared_error(y_test_scaled, nn_test_pred)
    print("Test Mean Squared Error:", mse_test)

    # Model Optimization
    # Lets try to optimize the parameters of the neural network model by using grid search shown.
    # The parameters to be optimized are hidden-layer sizes and activation functions
    # The mean square error is used as the metrics.
    # First, we need to import the GridSearchCV class and create a dict with the parameters as the
    # keys and potential values you want to try as the values:


async def main():
    print("--------------")
    explore_data()
    run_neural_network()


asyncio.run(main())