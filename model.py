import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
try:
    weather_data = pd.read_csv('weather_data.csv')
except FileNotFoundError:
    print("Error: 'weatherdata.csv' not found. Please upload the file.")
    exit()


# Data Preprocessing (example - adapt to your data)
# Handle missing values (replace with mean/median/mode or drop rows/columns)
weather_data = weather_data.fillna(weather_data.mean())

# Feature Engineering (example)
# You might need to create new features based on your data and domain expertise
# For example, create a 'day_of_year' feature, or combinations of existing features.
weather_data['day_of_year'] = pd.to_datetime(weather_data['Date']).dt.dayofyear

# Define features (X) and target (y)
# Replace 'Temperature', 'Humidity', etc. with your actual feature column names
features = ['Temperature', 'Humidity', 'WindSpeed'] # Example features, change accordingly
target = 'Temperature' # Example target, change accordingly


X = weather_data[features]
y = weather_data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model (RandomForestRegressor in this example)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction for new data
new_data = pd.DataFrame({'Temperature': [25], 'Humidity': [60], 'WindSpeed': [10]})
new_prediction = model.predict(new_data)
print(f"Prediction for new data: {new_prediction}")
