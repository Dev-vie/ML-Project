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
url = 'https://github.com/Dev-vie/ML-Project/blob/16dc61002041f707bacdd0fde76a3b8a81088077/Dataset/expenses_income_summary.csv'
df = pd.read_csv(url)
df.head()

#
