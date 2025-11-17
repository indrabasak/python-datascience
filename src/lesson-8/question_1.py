"""
Data Set Information
There are 10 predictors, all quantitative, and a binary dependent variable, indicating the
presence or absence of breast cancer. The predictors are anthropometric data and parameters
which can be gathered in routine blood analysis. Prediction models based on these predictors,
if accurate, can potentially be used as a biomarker of breast cancer.

Attribute Information
Quantitative	Attributes
Age	           (years)
BMI	           (kg/m2)
Glucose	       (mg/dL)
Insulin	       (µU/mL)
HOMA
Leptin	       (ng/mL)
Adiponectin	   (µg/mL)
Resistin	   (ng/mL)
MCP-1(pg/dL)   (ng/mL)

Labels
1 = Healthy Controls
2 = Patients

Task
1. Perform Data exploratory analysis on the data. (10 points)
2. Use 30% of data as the test set and build a Logistic regression model to predict Labels variable. (20 points)
3. Build the Naïve Bayes model to predict Labels variable. (20 points)
4. Build the Decision tree model to predict Labels variable. (20 points)
5. Build Neural network model to predict Labels variable. (20 points)
6. Which model is the best? Which variable is the most important one? (10 points)
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
# 1. Perform Data exploratory analysis on the data.
# ===============================================
breast_cancer = pd.read_csv("./data/lesson-8/breastcancer.csv")
print("Breast Cancer Dataset:")
print(breast_cancer.head())

print("Breast Cancer Dataset Description:")
print(breast_cancer.describe())

print("Missing Values in Breast Cancer Dataset:")
print(breast_cancer.isnull().sum().sum())

print("Breast Cancer Correlation Matrix:")
print(breast_cancer.corr())

print("Correlation Matrix Heatmap:")
plt.matshow(breast_cancer.corr())
plt.title('Correlation Matrix Heatmap', position=(0.5, 1.1))
plt.colorbar()
plt.xticks(range(10), list(breast_cancer.columns))
plt.yticks(range(10), list(breast_cancer.columns))
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# ===============================================
# 2. Use 30% of data as the test set and build a Logistic regression model to predict Labels variable.
# ===============================================
# This line selects columns 0 to 8 (inclusive) from the breast cancer DataFrame and assigns
# them to X. These columns represent the feature variables used for model training,
# excluding the target variable (Classification).
X = breast_cancer.iloc[:, 0 : 9]
y = breast_cancer.Classification

# Split the dataset into training and testing sets
# This line splits the dataset into training and test sets.
# X_train, y_train: 70% of the data, used to train models
# X_test, y_test: 30% of the data, used to evaluate models
# random_state=34 ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=34)

# Model Generation
lr = linear_model.LogisticRegression()
lr_fit = lr.fit(X_train, y_train)
print(lr_fit)
print("Coefficients:", lr_fit.coef_)
print("Intercept:", lr_fit.intercept_)

# Model Evaluation
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
# 3. Build the Naïve Bayes model to predict Labels variable.
# ===============================================
NB = GaussianNB()
NB_fit = NB.fit(X_train, y_train)
GaussianNB(priors=None)

# Model Evaluation
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

np.set_printoptions(precision=2)
NB.predict_proba(X_test[:5])

# ===============================================
# 4. Build the Decision tree model to predict Labels variable.
# ===============================================
DT = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5)
DT_fit = DT.fit(X_train, y_train)

# Model Evaluation
DT_pred = DT_fit.predict(X_test)
print("Decision Tree Testing Set Evaluation:")
print("Accuracy:", metrics.accuracy_score(y_test, DT_pred))
class_report = metrics.classification_report(y_test, DT_pred)
print("Classification Report:\n", class_report)

dt_importance = DT.feature_importances_
pd_df = pd.DataFrame({'variable': breast_cancer.columns[:9], 'importance': dt_importance})
print("Decision Tree Feature Importance:")
print(pd_df.sort_values(by='importance', ascending=False))

# ===============================================
# 5. Build Neural network model to predict Labels variable.
# ===============================================