📉 Customer Churn Prediction using Machine Learning
📌 Overview

Customer churn refers to customers discontinuing a company’s service. Retaining existing customers is significantly more cost-effective than acquiring new ones, making churn prediction a critical business problem.

This project builds an end-to-end machine learning pipeline to predict customer churn using structured customer data. The solution focuses on business-relevant evaluation metrics and proper handling of class imbalance.

🎯 Problem Statement

The objective is to predict whether a customer is likely to churn based on demographic, service usage, and billing information.

Since churn datasets are typically imbalanced, metrics such as Recall and ROC-AUC were prioritized over accuracy to better identify at-risk customers.

📊 Dataset

Each row represents a customer

Target variable: Churn

1 → Customer churned

0 → Customer retained

Feature types:

Numerical: tenure, MonthlyCharges, TotalCharges

Categorical: Contract, PaymentMethod, InternetService, etc.

🧹 Data Preprocessing

The following preprocessing steps were performed:

Handled missing values

Encoded categorical variables

Scaled numerical features where required

Split data into training and testing sets using a fixed random state for reproducibility

⚙️ Handling Class Imbalance

The dataset exhibited class imbalance, with churned customers forming a minority class.

To address this:

SMOTE (Synthetic Minority Over-sampling Technique) was applied only on the training data

This ensured improved model learning while preventing data leakage during evaluation

🤖 Models Trained

The following machine learning models were trained and evaluated:

Decision Tree – baseline tree-based model

Random Forest – ensemble bagging model

XGBoost – gradient boosting model

XGBoost was selected as the final model due to its superior performance in terms of ROC-AUC and recall.

📈 Model Evaluation
Metric	Value
Accuracy	78%
Recall (Churn = 1)	58%
Precision (Churn = 1)	59%
ROC-AUC	0.83
Evaluation Notes

Accuracy alone was not sufficient due to class imbalance

Recall and ROC-AUC were prioritized to effectively identify customers at risk of churning

The ROC-AUC score indicates strong discriminative ability between churned and retained customers

🔍 Model Explainability

Feature importance analysis revealed that the following factors strongly influence churn:

Contract type

Customer tenure

Monthly charges

These insights help bridge model predictions with actionable business decisions.

🧪 Sample Prediction

Example prediction for a single customer record:

Prediction: No Churn

Churn Probability: 17%

Probabilistic predictions allow businesses to define custom risk thresholds and prioritize retention strategies.

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

Matplotlib

🚀 Future Improvements

Decision threshold tuning to improve recall

Model deployment using FastAPI

SHAP-based model explainability

Integration with dashboards or downstream systems
