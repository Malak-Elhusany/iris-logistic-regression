# Iris Species Classification using Logistic Regression
A end-to-end Machine Learning pipeline for classifying Iris flower species using LogisticRegression. This project demonstrates production-grade data preprocessing, leakage-proof feature scaling, model evaluation, and inference on unseen data.

## 📌 Project Overview
The goal of this project is to accurately predict the species of an Iris flower (Setosa, Versicolor, or Virginica) based on four physical features:
Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)
Instead of treating species classification as a continuous regression problem, this pipeline implements Logistic Regression, a proper probabilistic classification algorithm well-suited for discrete categorical outcomes.

## 🛠️ Key Machine Learning Concepts Applied
1.Stratified Train-Test Split:
oUsed stratify=y during train_test_split to maintain identical target class distributions across both training ($80\%$) and testing ($20\%$) sets.
2.Prevention of Data Leakage (Scaler Fitting):
oImplemented StandardScaler strictly by applying .fit_transform() on X_train and only .transform() on X_test and new samples. This ensures evaluation metrics reflect model generalization to completely unseen distributions.
3.Multi-class Model Evaluation:
oEvaluated model predictions using accuracy_score alongside full classification_report metrics (Precision, Recall, and F1-Score) and a Confusion Matrix to measure per-class performance balance.
4.Single-sample Inference (predict_proba):
oTransformed raw single-sample inputs through the fitted scaling pipeline prior to outputting predicted class labels and prediction confidence probabilities.

## 📊 Model Results
The model achieved an overall accuracy of ~93.3% on the independent test dataset:
Class Metric	Precision	Recall	F1-Score	Support
0 (Setosa)	1.00	1.00	1.00	10
1 (Versicolor)	0.90	0.90	0.90	10
2 (Virginica)	0.90	0.90	0.90	10
Overall Accuracy	—	—	0.93	30

## 🚀 How to Run the Project
1. Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:
pip install numpy pandas scikit-learn
2. Execution Workflow
Run the code cells sequentially inside your Jupyter Notebook / Google Colab or executing as a Python script:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data & Preprocess
# Assume X (features) and y (target) are loaded
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Scale Features (Prevent Leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
lg = LogisticRegression(random_state=42)
lg.fit(X_train_scaled, y_train)

# 4. Model Evaluation
y_pred = lg.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Predict on New Sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)

print("Predicted Class:", lg.predict(new_sample_scaled)[0])
print("Class Probabilities:", lg.predict_proba(new_sample_scaled)[0])

📁 Repository Structure

├── iris.Logistic Regression.ipynb                 # Main Jupyter Notebook with full code & outputs

├── iris.csv                   # Dataset file

├── README.md                  # Project documentation & execution guide

