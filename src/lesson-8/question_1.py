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
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

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
lr_accuracy = metrics.accuracy_score(y_test, lr_test_pred)
print("Accuracy:", lr_accuracy)

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
nb_accuracy = metrics.accuracy_score(y_test, NB_test_pred)
print("Accuracy:", nb_accuracy)

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
dt_accuracy = metrics.accuracy_score(y_test, DT_pred)
print("Accuracy:", dt_accuracy)
class_report = metrics.classification_report(y_test, DT_pred)
print("Classification Report:\n", class_report)

dt_importance = DT.feature_importances_
importance_df = pd.DataFrame({'variable': breast_cancer.columns[:9], 'importance': dt_importance})
print("Decision Tree Feature Importance:")
print(importance_df.sort_values(by='importance', ascending=False))

# ===============================================
# 5. Build Neural network model to predict Labels variable.
# ===============================================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("First 3 rows of scaled training data:")
print(X_test_scaled[:3])

print("First 3 rows of scaled testing data:")
print(X_test_scaled[:3])

NN = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1)
NN.fit(X_train_scaled, y_train)

NN_pred = NN.predict(X_test_scaled)
print("Neural Network Testing Set Evaluation:")
nn_accuracy = metrics.accuracy_score(y_test, NN_pred)
print("Accuracy:", nn_accuracy)

# ==============================================
# 6. Which model is the best? Which variable is the most important one?
# ==============================================
print("Model Comparison on Testing Set:")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Naïve Bayes Accuracy: {nb_accuracy:.4f}")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

print("The best model is the one with the highest accuracy on the testing set.")
# The max function returns the largest item in an iterable or the largest of
# two or more arguments. You can provide a key parameter,
# which is a function that specifies a value to use for comparison.
# For example, in your code, max is used to find the model name with the highest
# accuracy by comparing values from a dictionary.
best_model = max(
    {
        "Logistic Regression": lr_accuracy,
        "Naïve Bayes": nb_accuracy,
        "Decision Tree": dt_accuracy,
        "Neural Network": nn_accuracy,
    },
    key=lambda k: {
        "Logistic Regression": lr_accuracy,
        "Naïve Bayes": nb_accuracy,
        "Decision Tree": dt_accuracy,
        "Neural Network": nn_accuracy,
    }[k],
)
print(f"The best model is: {best_model}")

# Identify the most important variable from the Decision Tree model
importance_df = pd.DataFrame({'variable': breast_cancer.columns[:9], 'importance': dt_importance})
sorted_importance = importance_df.sort_values(by='importance', ascending=False)
most_important_variable = sorted_importance.iloc[0]
print(f"The most important variable is: {most_important_variable['variable']} with importance {most_important_variable['importance']:.4f}")