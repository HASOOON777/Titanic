import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ==========================================
# 1. Data Loading & EDA (Exploratory Data Analysis)
# ==========================================
print("Loading data...")
try:
    df = pd.read_csv('train.csv')
    print("Data loaded locally successfully!")
except FileNotFoundError:
    print("Local train.csv not found, downloading from internet...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print("Data downloaded successfully!")

print("-" * 50)
print("1. Exploratory Data Analysis (EDA)")
print("Dataset Summary:")
print(df.describe())

# Save a heatmap for missing values (Outliers/Missing visually)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig("missing_values_heatmap.png")
print("-> Saved missing values heatmap to 'missing_values_heatmap.png'")
print("-" * 50)

# ==========================================
# 2. Data Cleaning & Preprocessing
# ==========================================
print("2. Preprocessing Data...")

# A. Imputation for Age (Using Median to avoid outliers impact)
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']])

# B. Imputation for Embarked (Using Most Frequent)
embarked_imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = embarked_imputer.fit_transform(df[['Embarked']]).ravel()

# C. Drop columns with too many missing values or irrelevant data
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

# D. Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.drop(['Name', 'PassengerId'], axis=1, inplace=True)

# E. Encoding (Converting categorical to numerical)
# Using LabelEncoder for 'Sex'
sex_encoder = LabelEncoder()
df['Sex'] = sex_encoder.fit_transform(df['Sex'])

# Using Pandas get_dummies for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# F. Feature Scaling (StandardScaler)
print("Applying StandardScaler...")
scaler = StandardScaler()

# Separating Features (X) and Target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Save column names to ensure consistency later
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')

# Scale the features
X_scaled = scaler.fit_transform(X)

print("Preprocessing completed successfully!")
print("-" * 50)

# ==========================================
# 3. Model Building (Logistic Regression)
# ==========================================
print("3. Training Logistic Regression Model...")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Model training finished!")
print("-" * 50)

# ==========================================
# 4. Evaluation
# ==========================================
print("4. Model Evaluation:")

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died (0)', 'Survived (1)'], 
            yticklabels=['Died (0)', 'Survived (1)'])

plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("-> Saved confusion matrix to 'confusion_matrix.png'")
print("-" * 50)

# ==========================================
# 5. Deployment Readiness (Saving Models)
# ==========================================
print("5. Saving models and transformers for deployment...")

joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(age_imputer, 'age_imputer.pkl')
joblib.dump(sex_encoder, 'sex_encoder.pkl')

print("Success! All objects saved as .pkl files. Ready for Streamlit Deployment!")