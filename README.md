# Iris Species Classification with Logistic Regression

A clean, end-to-end Machine Learning project classifying Iris flower species (0, 1, 2) using Logistic Regression in scikit-learn, following proper data-splitting, scaling, and evaluation practices.

Project Overview
Bullet Task: Multi-class classification of Iris species.
Bullet Model: LogisticRegression (scikit-learn).
Bullet Preprocessing: Stratified train-test split and StandardScaler fit strictly on training data (fit_transform) and applied to test and new samples (transform) to prevent data leakage.
Bullet Evaluation: Evaluated using Accuracy, Confusion Matrix, and Classification Report (precision, recall, f1-score, support).
Bullet Inference: Includes predicting class labels (predict) and class probabilities (predict_proba) on unseen samples.

Workflow and Key Practices

1. Data Loading and Preprocessing: Loaded the Iris dataset, split features (X) and target (y), and applied stratified splitting before scaling.

2. Leak-Free Scaling: Fit StandardScaler on X_train only, then transformed X_test and any new unseen samples using the fitted scaler.

3. Model Training: Trained a LogisticRegression classifier on the scaled training set.

4. Model Evaluation: Assessed performance using accuracy_score, confusion_matrix, and classification_report rather than correlation matrices.

5. Unseen Sample Inference: Scaled new input samples with .transform() and inspected both predicted classes (.predict()) and class probabilities (.predict_proba()).

Tech Stack
Bullet Python
Bullet Pandas and NumPy
Bullet Scikit-learn
Bullet Jupyter Notebook / VS Code

How to Run

1. Open the project notebook in Jupyter Notebook or VS Code.

2. Run the cells sequentially from data loading and preprocessing through training, evaluation, and sample prediction.

3. Save the notebook and upload the project files along with README.md to GitHub.
