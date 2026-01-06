# Customer Churn Prediction using Machine Learning

## Overview
Customer churn refers to customers discontinuing a company’s service. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project builds an end-to-end machine learning pipeline to predict customer churn using structured customer data, with a focus on business-relevant evaluation metrics and proper handling of class imbalance.

## Problem Statement
The objective is to predict whether a customer is likely to churn based on demographic, service usage, and billing information. Since churn datasets are typically imbalanced, recall and ROC-AUC were prioritized over accuracy to better identify at-risk customers.

## Dataset
Each row represents a customer record.

Target variable:
- Churn = 1 → Customer churned
- Churn = 0 → Customer retained

Feature types include:
- Numerical features such as tenure, MonthlyCharges, and TotalCharges
- Categorical features such as Contract, PaymentMethod, and InternetService

## Data Preprocessing
The following preprocessing steps were applied:
- Handling missing values
- Encoding categorical variables
- Scaling numerical features where required
- Splitting the dataset into training and testing sets using a fixed random state for reproducibility

## Handling Class Imbalance
The dataset exhibited class imbalance, with churned customers forming a minority class. SMOTE (Synthetic Minority Over-sampling Technique) was applied only on the training data to balance the classes while preventing data leakage during evaluation.

## Models Trained
The following machine learning models were trained and evaluated:
- Decision Tree
- Random Forest
- XGBoost

XGBoost was selected as the final model due to its superior performance in terms of recall and ROC-AUC.

## Model Evaluation
The final model achieved the following results:
- Accuracy: 78%
- Recall (Churn = 1): 58%
- Precision (Churn = 1): 59%
- ROC-AUC: 0.83

Accuracy alone was not sufficient due to class imbalance. Recall and ROC-AUC were prioritized to better identify customers at risk of churning.

## Model Explainability
Feature importance analysis showed that contract type, customer tenure, and monthly charges were among the strongest predictors of customer churn. These insights help connect model predictions with actionable business decisions.

## Sample Prediction
Example prediction for a single customer:
- Prediction: No Churn
- Churn Probability: 17%

Probabilistic outputs allow businesses to define custom risk thresholds and prioritize retention efforts.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib

## Future Improvements
- Decision threshold tuning to further improve recall
- Deployment using FastAPI for real-time predictions
- SHAP-based explainability for deeper insights
- Integration with dashboards or downstream systems

## How to Run
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the notebook:
   jupyter notebook

## Key Takeaways
- Built a complete machine learning pipeline from preprocessing to evaluation
- Addressed class imbalance using SMOTE
- Focused on business-relevant evaluation metrics
- Produced explainable and probabilistic predictions
