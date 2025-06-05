# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load Dataset
from google.colab import files # I used Google Colab to host my code
uploaded = files.upload()
df = pd.read_csv('expenses_income_summary.csv')
df.head()

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing data
df = df.drop(columns=["transfer-amount", "transfer-currency","to-account","receive-amount","receive-currency","description","due-date"])
df_expense = df[df['type'] == 'EXPENSE'].dropna(subset=['title', 'category','amount'])
print(df_expense.head())
print(df_expense.isnull().sum())

# Save cleaned dataset
df_expense.to_csv('cleaned_expenses.csv', index=True)
files.download('cleaned_expenses.csv')
# Feature
X = df_expense['title'] # The input feature
y = df_expense['category'] # The target class

# Convets Text to Vectors (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_grids = {
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'solver': ['saga'],
        'multi_class': ['multinomial'],
        'class_weight': ['balanced'],
        'max_iter': [1000]
    },
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'class_weight': ['balanced'],
        'random_state': [42]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf'],
        'class_weight': ['balanced'],
        'random_state': [42]
    }
}

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Models Training & Evaluation
best_models = {}

for name in models:
    print(f"\nTuning {name}...")
    grid = GridSearchCV(models[name], param_grids[name], cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    
    print(f"Best Parameters for {name}: {grid.best_params_}")
    
    # Evaluate on test set
    y_pred = best_models[name].predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_models[name].classes_)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=best_models[name].classes_,
                yticklabels=best_models[name].classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
