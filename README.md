# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
<br>
# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')
<br>
# Step 1: Data Exploration
print(titanic_data.info())
print(titanic_data.describe())
print(titanic_data.isnull().sum())
<br>
# Step 2: Data Preprocessing
# Drop unnecessary columns (PassengerId, Name, Ticket, and Cabin have less impact on survival)
titanic_data = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
<br>
# Handle missing values for Age and Embarked
imputer = SimpleImputer(strategy='mean')
titanic_data['Age'] = imputer.fit_transform(titanic_data[['Age']])
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
<br>
# Convert categorical features (Sex, Embarked) into numeric
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'])
<br>
# Step 3: Data Visualization (optional)
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(titanic_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
<br>
# Step 4: Split the data into training and testing sets
X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
<br>
# Step 5: Model Selection (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
<br>
# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
<br>
# Classification report
print(classification_report(y_test, y_pred))
<br>
# Optional: Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title('Feature Importance')
plt.show()
