# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
<br>
# Load the dataset
# Make sure to replace 'path_to_your_dataset.csv' with the actual path to your dataset
df = pd.read_csv('path_to_your_dataset.csv')
<br>
# Display the first few rows of the dataset
print(df.head())
<br>
# Check for class imbalance
print("Class distribution:")
print(df['Class'].value_counts())
<br>
# Visualize the class distribution
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class (0: Genuine, 1: Fraud)')
plt.ylabel('Count')
plt.show()
<br>
# Split the dataset into features and target variable
X = df.drop('Class', axis=1)  # Features
y = df['Class']                # Target variable
<br>
# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
<br>
# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
<br>
# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
<br>
# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
<br>
# Make predictions
y_pred_logistic = logistic_model.predict(X_test)
<br>
# Evaluate the Logistic Regression model
print("Logistic Regression Model Evaluation:")
print(classification_report(y_test, y_pred_logistic))
<br>
# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
<br>
# Make predictions
y_pred_rf = rf_model.predict(X_test)
<br>
# Evaluate the Random Forest model
print("Random Forest Model Evaluation:")
print(classification_report(y_test, y_pred_rf))
<br>
# Confusion Matrix for Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'])
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
