# Credit Card Default Prediction

## Overview
This repository contains code for building and evaluating machine learning models to predict credit card defaults. The dataset used for this analysis contains information about credit card holders, including demographics, payment history, and bill statements. The goal is to develop models that can effectively predict whether a credit card holder will default on their payments.

## Dataset
The dataset used for this analysis is the Credit Card Default dataset, which contains 30,000 records and 24 features. These features include demographic information, payment history, and bill amounts for six months.

## Part 1: Exploratory Data Analysis (EDA)
The EDA process involves exploring the dataset to understand its structure and characteristics. Some key steps in the EDA process include:
- Checking for missing values and handling them appropriately.
- Exploring the distribution of the target variable (default/non-default).
- Visualizing the distribution of features to understand their impact on the target variable.
- Performing correlation analysis to identify relationships between features.

## Part 2: Machine Learning Models
In this part, we build and evaluate machine learning models to predict credit card defaults. We use three classification models: Logistic Regression, Decision Tree, and Random Forest Classifier. The steps involved in building and evaluating these models are as follows:

### Data Preprocessing
- Splitting the data into training and testing sets.
- Standardizing the features to ensure they have a similar scale.

### Feature Selection
- Using Recursive Feature Elimination (RFE) to select the most important features for modeling.

### Model Building
- Training the models on the training set.
- Tuning hyperparameters using RandomizedSearchCV to improve model performance.

### Model Evaluation
- Evaluating the models using metrics such as accuracy, precision, recall, and F1-score.
- Using 5-fold cross-validation to assess the models' generalization performance.

### Result Analysis
- Comparing the performance of different models using various metrics.
- Visualizing the ROC curves and confusion matrices to understand model performance.

## Conclusion
- The Random Forest Classifier outperforms other models with an accuracy of 0.82 and an AUC of 0.77.
- Feature importance analysis reveals that repayment status, bill amounts, and previous payment amounts are key predictors of credit card defaults.
- The analysis highlights the importance of using appropriate metrics and techniques for evaluating models on imbalanced datasets.

## Future Work
- Further tuning the models to improve performance.
- Exploring additional feature engineering techniques to enhance model predictive power.
- Investigating the use of ensemble methods or advanced algorithms for better results.

## Repository Structure
- `data`: Contains the dataset used for analysis.
- `notebooks`: Jupyter notebooks for EDA, feature selection, model building, and evaluation.
- `images`: Contains visualizations generated during the analysis.
- `README.md`: Provides an overview of the project and its components.

## Dependencies
- Python 3
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
