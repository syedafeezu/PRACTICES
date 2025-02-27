import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# Load the dataset
data = pd.read_csv('weather_data.csv')

# Drop unnecessary columns
data = data.drop(['date', 'date_id', 'day_date', 'day_name', 'Start_hour', 'End_hour'], axis=1)

# Handle missing values (if any)
data = data.fillna(data.mean())

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['desc'], drop_first=True)

# Separate features and target variable
X = data.drop('temp', axis=1)
y = data['temp']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest Mean Squared Error: {mse_rf}')


# Initialize the XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost Mean Squared Error: {mse_xgb}')

# Compare the performance of the two models
if mse_rf < mse_xgb:
    print("Random Forest performs better.")
else:
    print("XGBoost performs better.")
    
    
import joblib

# Save the best model
if mse_rf < mse_xgb:
    joblib.dump(rf_model, 'best_weather_model.pkl')
else:
    joblib.dump(xgb_model, 'best_weather_model.pkl')
    
# Load the model
model = joblib.load('best_weather_model.pkl')

# Example: Predict temperature for new data
new_data = np.array([[29.0, 1002.0, 80, 7, 15, 2024, 0]])  # Example input
new_data = scaler.transform(new_data)  # Standardize the new data
predicted_temp = model.predict(new_data)
print(f'Predicted Temperature: {predicted_temp[0]}')