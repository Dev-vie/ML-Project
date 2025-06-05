# Smart Expense Tracker 🧾

This machine learning project classifies expense descriptions into categories using Logistic Regression, Random Forest, and SVM. It helps automate and improve financial tracking.

## 💡 Features
- Classifies text-based expense entries (e.g., "Burger", "Grab")
- Supports hyperparameter tuning with Grid Search
- Includes TF-IDF preprocessing
- Evaluates accuracy and generates confusion matrices

## 📊 Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## 🧪 Dataset
- Source: Kaggle
- File: `expenses_income_summary.csv`
- Filtered to include only `type = 'EXPENSE'`

## 🛠️ Setup
```bash
pip install pandas scikit-learn matplotlib seaborn
