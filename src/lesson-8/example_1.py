"""
In lesson 8, I will introduce four classification models with sklearn package.

Sklearn provides a standard way to build and evaluate models. Once you learn one,
it is easy to transfer to other models.

In this video, I will introduce logistic regression. First, let’s import the necessary
packages: pandas, numpy, matplotlib, os, and import
train_test_split and metrics from sklearn.

For lesson 8, we will use glass as the example. Here we change working directory and load the
data into Python. The data doesn’t have a column name. So we need to assign the column names to it.
ID is irrelevant to data mining task, we need to drop it.

We can explore the data, this is the head, statistic analysis, use value_Counts to check the
categories and distribution. If there are missing values.
Correlation of variables, You can try to plot the correlation matrix using matshow.
Adding a title, color bar, change x, y ticks and x, y labels.

Next, we need to split the data into train and test, this has been introduced in lesson 7.
Here we use 30 percent as a test set.

LogesticRegression model is under sklean linear_model, we can import linear_model from sklean.

Line 50 is used to initiate the model, if you want to change argument, you need to do it here.

Now, if you type “lr.” and press “Tab”, all functions associated with lr will be listed here.
There are very similar to all models. Fit is used to build the model with supplied data.

Line 51 is used X_Train and y_train to build the model.

For the linear model, you can use coef_ and intercept_ to check the coefficients and intercept.

No you can evaluate your model on both train and test set.

Predict function will predict the class for given data. Here we predict the class label for
both train and test data, you can evaluate the accuracy by using metrics.accuray_score by providing
the true label and predicted label.

Line 60-63 are confusion matrices for both train and test data.
You can also use matshow to plot confusion matrix.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# ===============================================
# Data Preparation
# ===============================================
glass = pd.read_csv("./data/lesson-8/glass+identification/glass.data", header=None)
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
glass = glass.drop('Id', axis=1) # Drop ID column as it's unnecessary for analysis
print("Glass Dataset:")
print(glass.head())

print("Glass Dataset Description:")
print(glass.describe())

print("Glass Type Value Counts:")
print(glass.Type.value_counts())

print("Missing Values in Glass Dataset:")
print(glass.isnull().sum().sum())

print("Correlation Matrix:")
print(glass.corr())

print("Correlation Matrix Heatmap:")
plt.matshow(glass.corr())
plt.title('Correlation Matrix Heatmap', position=(0.5, 1.1))
plt.colorbar()
plt.xticks(range(10), list(glass.columns))
plt.yticks(range(10), list(glass.columns))
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Split the dataset into training and testing sets
X = glass.iloc[:, 0 : 9]
print(X)
y = glass.Type
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=34)

# ===============================================
# Model Generation
# ===============================================
lr = linear_model.LogisticRegression()
lr_fit = lr.fit(X_train, y_train)
print(lr_fit)
print("Coefficients:", lr_fit.coef_)
print("Intercept:", lr_fit.intercept_)

# ===============================================
# Model Evaluation
# ===============================================
lr_train_pred = lr_fit.predict(X_train)
lr_test_pred = lr_fit.predict(X_test)
print("Training Set Evaluation:")
print("Accuracy:", metrics.accuracy_score(y_train, lr_train_pred))
print("Testing Set Evaluation:")
print("Accuracy:", metrics.accuracy_score(y_test, lr_test_pred))

train_cm = metrics.confusion_matrix(y_train, lr_train_pred)
print("Training Confusion Matrix:")
print(train_cm)
test_cm = metrics.confusion_matrix(y_test, lr_test_pred)
print("Testing Confusion Matrix:")
print(test_cm)

plt.figure()
plt.matshow(test_cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# ===============================================
# Lesson 8.2
# ===============================================
# Naïve Bayes
#
# The second model we introduce is Naïve Bayes, and we still use the glass data.
# Once you import the model, the rest is similar to how to build and evaluate the
# Logistic Regression Model
# ===============================================

# ==============================================
# Model Generation
# GaussianNB is one of the Naïve Bayes models in scikit=learn. You can crate a GaussianNB object,
# then fit it with train datasets (see Figure 8.13).
#` ==============================================
NB = GaussianNB()
NB_fit = NB.fit(X_train, y_train)
GaussianNB(priors=None)

# For the Naïve Bayes model, it only has one parameter:
# Priors: Priori probabilities of the classes.
# Since this information of the classes is unknown, we don’t specify this parameter in the model.

# ==============================================
# Model Evaluation
# The model evaluation process is similar to what you did in lesson 8.1.
# You need to predict the target labels for both train and test data, and then calculate their
# accuracy and confusion matrix (see Figures 8.14 A, B, & C):
# ==============================================
NB_train_pred = NB_fit.predict(X_train)
NB_test_pred = NB_fit.predict(X_test)
print("Naïve Bayes Training Set Evaluation:")
print("Accuracy:", metrics.accuracy_score(y_train, NB_train_pred))
print("Naïve Bayes Testing Set Evaluation:")
print("Accuracy:", metrics.accuracy_score(y_test, NB_test_pred))

print("Naïve Bayes Training Confusion Matrix:")
train_cm = metrics.confusion_matrix(y_train, NB_train_pred)
print(train_cm)
print("Naïve Bayes Testing Confusion Matrix:")
test_cm = metrics.confusion_matrix(y_test, NB_test_pred)
print(test_cm)

# The Bayesian model can also provide the probability to each class instead of final labels.
# Figure 8.15 shows how to compute the probabilities for the top 5 rows of
# test data by using the predict_proba method:
# returns the predicted probabilities for each class for the first 5 samples
# in X_test using the trained Naïve Bayes model (NB). Instead of just predicting the
# class label, it gives the likelihood of each sample belonging to each possible class.
# The output is a 2D array where each row corresponds to a sample and each column to a class.
np.set_printoptions(precision=2)
NB.predict_proba(X_test[:5])

# ===============================================
# Lesson 8.3
# ===============================================
# Decision Tree
# There are various decision tree algorithms such as ID3, C4.5, C5.0, and CART.
# scikit-learn provides an optimized version of the CART model.
DT = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5)
DT_fit = DT.fit(X_train, y_train)

# Model Evaluation
# Next, we predict the glass type and evaluate the performance of the test set.
# This time we use a classification report which shows the model performance on
# each type and the overall performance
DT_pred = DT_fit.predict(X_test)
print("Decision Tree Testing Set Evaluation:")
print("Accuracy:", metrics.accuracy_score(y_test, DT_pred))
class_report = metrics.classification_report(y_test, DT_pred)
print("Classification Report:\n", class_report)

# Decision tree model also provides an attribute to return the feature importance.
# Since the method returns an array, we create a DataFrame to show both the
# variable names and importance factors
dt_importance = DT.feature_importances_
pd_df = pd.DataFrame({'variable': glass.columns[:9], 'importance': dt_importance})
print("Decision Tree Feature Importance:")
print(pd_df.sort_values(by='importance', ascending=False))