# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
<br>
# Load the dataset
url = 'https://raw.githubusercontent.com/yourusername/yourrepository/main/sales_data.csv'  # Replace with the actual dataset URL
df = pd.read_csv(url)
<br>
# Display the first few rows of the dataset
print(df.head())
<br>
# Check for missing values
print(df.isnull().sum())
<br>
# Data Visualization
sns.scatterplot(data=df, x='Advertising Spend', y='Sales')
plt.title('Advertising Spend vs Sales')
plt.xlabel('Advertising Spend')
plt.ylabel('Sales')
plt.show()
<br>
# Define features and target variable
X = df[['Advertising Spend']]
y = df['Sales']
<br>
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
<br>
# Create a linear regression model
model = LinearRegression()
<br>
# Train the model
model.fit(X_train, y_train)
<br>
# Make predictions
y_pred = model.predict(X_test)
<br>
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
<br>
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
<br>
# Plotting the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Sales')
plt.title('Sales Prediction')
plt.xlabel('Advertising Spend')
plt.ylabel('Sales')
plt.legend()
plt.show()
