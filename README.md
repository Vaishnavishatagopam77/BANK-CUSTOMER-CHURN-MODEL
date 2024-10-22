# BANK-CUSTOMER-CHURN-MODEL
# Project Overview
This project involves building a predictive model using Support Vector Machine (SVM) to determine the likelihood of a customer leaving a bank (churning). SVM is known for its ability to handle classification problems with high-dimensional data, and it works by finding the optimal hyperplane that separates the classes (churn or not churn).
# Problem Statement
Customer churn is a significant issue for banks, leading to loss of revenue and increased customer acquisition costs. The goal of this project is to use SVM to predict whether a customer will churn, allowing the bank to identify and retain at-risk customers.
# Dataset
The dataset includes customer information such as demographics, account details, and banking activity. The target variable is Churn, which indicates whether the customer has left the bank.
# Features.
Customer ID
Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Has Credit Card
Is Active Member
Estimated Salary
Churn (0 = No, 1 = Yes).
# Installation.
google colab file .ipynb.
# Requirements
Python 3.7+

Scikit-learn

Pandas

NumPy

Matplotlib
# Model Development
1. Data Preprocessing

Handling missing values: Checking for and imputing missing data.
Feature Encoding: Convert categorical variables like Gender and Geography into numerical values using One-Hot Encoding.
Scaling: Since SVMs are sensitive to the range of input data, standard scaling is applied to features like Credit Score, Balance, and Estimated Salary.
2. Support Vector Machine (SVM)
The SVM model is selected for its effectiveness in binary classification problems. Here's the process:

* Kernel Selection: We use the RBF kernel (Radial Basis Function), which handles non-linear decision boundaries well. Other kernels such as linear or polynomial can be explored.

3.Hyperparameter Tuning: The C parameter (controls the trade-off between correct classification and margin maximization) and gamma (kernel coefficient) are tuned using Grid Search.

4. Cross-Validation
We use k-fold cross-validation to ensure the model generalizes well on unseen data.

5. Model Pipeline
We create a pipeline that includes preprocessing, scaling, and applying the SVM classifier to streamline the process.

# Evaluation Metrics
The performance of the model is evaluated using various metrics:

Accuracy: Proportion of correctly classified instances.

Precision: How many of the predicted churns were actual churns.

Recall: How many of the actual churn cases were correctly predicted.

F1-Score: The harmonic mean of precision and recall, balancing both metrics.

ROC-AUC: Measures the ability of the model to distinguish between churn and non-churn cases.

ROC-AUC Score: Measures the model's ability to distinguish between classes.
# Results.
After training and tuning the SVM model, the best performance achieved was:
# For Model Accuracy simple one.
Accuracy: 84%

Precision: 83%

Recall: 62%

F1-Score: 84%


# For Model with Random Under Sampling
Accuracy: 73%

Precision: 73%

Recall: 75%

F1-Score: 74%


# For Model with Random Over Sampling
Accuracy: 75%

Precision: 74%

Recall: 77%

F1-Score: 76%
