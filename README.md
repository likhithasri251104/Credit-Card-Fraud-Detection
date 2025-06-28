# 💳 AI-Based Credit Card Fraud Detection System

A machine learning project for detecting fraudulent credit card transactions using classification algorithms. This solution uses real-world, imbalanced data and incorporates various techniques including preprocessing, resampling (SMOTE), and model evaluation.

---

## 📌 Project Overview

Credit card fraud is a significant challenge in financial security. This project utilizes supervised machine learning to detect fraudulent transactions. It involves:

- Data cleaning and preprocessing
- Handling imbalanced datasets using SMOTE
- Feature analysis and visualization
- Model training with classifiers such as Logistic Regression, KNN, XGBoost, Decision Tree, and Random Forest
- Hyperparameter tuning and performance evaluation

---

## 📁 Dataset

This project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) which contains transactions made by European cardholders in September 2013.

- **Rows:** 284,807
- **Features:** 30 (V1-V28 are PCA components, plus `Time`, `Amount`, and `Class`)
- **Class:** 0 for non-fraud, 1 for fraud

---

## 📂 Project Structure

fraud-detection/
├── data/
│ └── creditcard.csv
├── notebooks/
│ └── credit_card_fraud_detection.ipynb
├── models/
│ └── trained_models.pkl
├── README.md
└── requirements.txt



---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Clone the repository

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection



📊 Features Covered
🧼 Data Cleaning: Removal of null values and duplicates.

📈 Exploratory Data Analysis (EDA): Histograms, box plots, correlation heatmaps.

⚖️ Imbalance Handling: SMOTE (Synthetic Minority Over-sampling Technique).

📦 Feature Scaling: Using standardization and normalization.

🧠 Model Training: Multiple classifiers (Logistic Regression, KNN, Decision Tree, XGBoost, Random Forest).

🔍 Model Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

🔧 Hyperparameter Tuning: GridSearchCV for optimal performance.

🧪 Model Performance
Each classifier is evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

The model that performs best in terms of Recall and F1-Score is typically chosen for production use.

📉 Sample Output (Confusion Matrix)

Confusion Matrix for XGBoost:
[[85267    33]
 [   19   123]]
Accuracy: 0.9994
Precision: 0.7885
Recall: 0.8662
F1 Score: 0.8257
📈 Visualizations
Correlation Heatmaps

Feature importance

Class Distribution

Boxplots for Outlier Detection

Confusion Matrices for each model

🧠 Models Used
✅ Logistic Regression

✅ K-Nearest Neighbors

✅ Decision Tree

✅ Random Forest

✅ XGBoost

📊 Libraries & Tools
pandas, numpy – Data Manipulation

matplotlib, seaborn – Data Visualization

scikit-learn – Machine Learning models and tools

xgboost – Gradient boosting framework

imblearn – SMOTE for imbalance handling

