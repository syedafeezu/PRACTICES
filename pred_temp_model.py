import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load dataset
try:
    df = pd.read_csv("weather_forecast.csv")
except FileNotFoundError:
    raise FileNotFoundError("The dataset file 'weather_forecast.csv' was not found.")

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Outlook", "Humidity", "Windy", "Play", "Temperature"]

for column in categorical_columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Define features and target
if "Temperature" not in df.columns:
    raise KeyError("The dataset does not contain a 'Temperature' column.")

X = df.drop(columns=["Temperature"])
y = df["Temperature"]

# Check for NaN values
df.fillna(df.mean(), inplace=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Function to predict temperature based on user input
def predict_temperature(outlook, humidity, windy, play):
    try:
        input_data = pd.DataFrame([[outlook, humidity, windy, play]], columns=["Outlook", "Humidity", "Windy", "Play"])
        for column in input_data.columns:
            if column in label_encoders:
                if input_data[column][0] in label_encoders[column].classes_:
                    input_data[column] = label_encoders[column].transform([input_data[column][0]])
                else:
                    input_data[column] = -1  # Assign a default value for unseen labels
        predicted_temp = model.predict(input_data)[0]
        print(f"Predicted Temperature: {predicted_temp:.2f}")
    except Exception as e:
        print(f"Error in prediction: {e}")

# Example usage
predict_temperature("Sunny", "High", "False", "Yes")
predict_temperature("Overcast", "Normal", "True", "No")