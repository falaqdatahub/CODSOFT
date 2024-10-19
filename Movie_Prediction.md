# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
<br>
# Load the dataset
df = pd.read_csv('imdb_india_movies.csv')
<br>
# Handle missing values
df.dropna(inplace=True)
<br>
# One-hot encode categorical variables
encoder = OneHotEncoder()
genre_encoded = encoder.fit_transform(df[['Genre']])
director_encoded = encoder.fit_transform(df[['Director']])
actors_encoded = encoder.fit_transform(df[['Actors']])
<br>
# Concatenate encoded variables with the original dataframe
df_encoded = pd.concat([df, pd.DataFrame(genre_encoded.toarray()), pd.DataFrame(director_encoded.toarray()), pd.DataFrame(actors_encoded.toarray())], axis=1)
<br>
# Drop original categorical variables
df_encoded.drop(['Genre', 'Director', 'Actors'], axis=1, inplace=True)
<br>
# Scale numerical variables
scaler = StandardScaler()
df_encoded[['Rating']] = scaler.fit_transform(df_encoded[['Rating']])
<br>
# Split data into training and testing sets
X = df_encoded.drop(['Rating'], axis=1)
y = df_encoded['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
<br>
# Build the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
<br>
# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
