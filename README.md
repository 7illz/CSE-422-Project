# Telco Customer Churn Prediction

## Overview

This repository contains a comprehensive Machine Learning and Deep Learning pipeline designed to predict customer churn in the telecommunications industry. By analyzing customer demographics, account information, and service usage, this project identifies patterns that lead to customer attrition, enabling proactive retention strategies.

## Dataset

The project utilizes the **Telco Customer Churn dataset**, which contains 7,043 customer records and 21 attributes.

* **Target Variable:** `Churn` (Yes/No)
* **Class Distribution:** 5,174 Retained (No) vs. 1,869 Churned (Yes)
* **Features:**
* **Demographics:** Gender, Senior Citizen status, Partner, Dependents.
* **Account Info:** Tenure, Contract type, Paperless Billing, Payment Method, Monthly Charges, Total Charges.
* **Services Subscribed:** Phone, Multiple Lines, Internet (DSL/Fiber), Online Security, Device Protection, Tech Support, Streaming TV/Movies.



## Data Preprocessing & Feature Engineering

To prepare the raw data for modeling, the following preprocessing pipeline was implemented:

1. **Data Cleaning:** Handled missing values in the `TotalCharges` column by coercing errors to numeric and dropping the resulting nulls (11 rows removed). Dropped the non-predictive `customerID` column.
2. **Categorical Encoding:**
* Applied `LabelEncoder` to binary categorical features (e.g., gender, Partner, PaperlessBilling).
* Applied One-Hot Encoding (`pd.get_dummies`) to multi-class features.


3. **Feature Scaling:** Standardized numerical variables (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to ensure zero mean and unit variance.
4. **Dimensionality Reduction (Multicollinearity):** Analyzed correlation heatmaps and removed redundant dummy variables (e.g., `InternetService_No`, `OnlineSecurity_No internet service`, `MultipleLines_No phone service`) to prevent multicollinearity and simplify the feature space.

## Model Architectures & Experiments

Three distinct classification models were trained and evaluated to find the best-performing architecture:

1. **K-Nearest Neighbors (KNN):** Configured with `n_neighbors=5` as a baseline spatial model.
2. **Logistic Regression:** Configured with `max_iter=1000` to serve as a robust, interpretable linear baseline.
3. **Artificial Neural Network (ANN):** A TensorFlow/Keras Sequential model consisting of:
* Input layer mapping to the processed feature space.
* Hidden Layer 1: 32 neurons, ReLU activation.
* Hidden Layer 2: 16 neurons, ReLU activation.
* Output Layer: 1 neuron, Sigmoid activation (Binary Crossentropy loss).



## Results & Evaluation

Models were evaluated based on Accuracy, Precision, Recall, and ROC-AUC scores.

| Model | Test Accuracy | Precision (Churn) | Recall (Churn) |
| --- | --- | --- | --- |
| **Logistic Regression** | **80.52%** | **0.65** | **0.57** |
| Artificial Neural Network | 79.57% | *varies* | *varies* |
| K-Nearest Neighbors | 76.30% | 0.56 | 0.55 |

**Key Insights:**

* **Best Performer:** **Logistic Regression** achieved the highest overall accuracy (80.52%) and provided the best balance of Precision and Recall, proving highly effective for this tabular dataset.
* The **Artificial Neural Network** performed competitively (~79.6%) but did not out-perform the simpler Logistic Regression model, suggesting the feature relationships are predominantly linear.
* Visualizations including Confusion Matrices, ROC Curves, and Bar Charts comparing Precision/Recall across all models are included in the notebook to provide a granular view of true positive vs. false positive tradeoffs.

## Technologies Used

* **Python 3.x**
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (StandardScaler, LabelEncoder, LogisticRegression, KNeighborsClassifier)
* **Deep Learning:** `TensorFlow`, `Keras`
* **Data Visualization:** `matplotlib`, `seaborn`
