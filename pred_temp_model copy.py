import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load dataset
df = pd.read_csv("weather_forecast.csv")

# Convert Temperature to numeric if necessary
temp_mapping = {"Cold": 10, "Warm": 20, "Hot": 30}  # Example mapping
if df["Temperature"].dtype == 'object':
    df["Temperature"] = df["Temperature"].map(temp_mapping)

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Outlook", "Humidity", "Windy", "Play"]

for column in categorical_columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Ensure 'Temperature' is numerical and scale it
scaler = MinMaxScaler()
df["Temperature"] = scaler.fit_transform(df[["Temperature"]].astype(float))

X = df.drop(columns=["Temperature"])
y = df["Temperature"]

# Check for NaN values
df.fillna(df.mean(numeric_only=True), inplace=True)

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
# def predict_temperature(outlook, humidity, windy, play):
#     input_data = pd.DataFrame([[outlook, humidity, windy, play]], columns=["Outlook", "Humidity", "Windy", "Play"])
    
#     for column in input_data.columns:
#         if column in label_encoders:
#             if input_data[column][0] in label_encoders[column].classes_:
#                 input_data[column] = label_encoders[column].transform([input_data[column][0]])
#             else:
#                 input_data[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])  # Assign default known class
    
#     predicted_temp_scaled = model.predict(input_data)[0]
#     predicted_temp = scaler.inverse_transform(np.array([[predicted_temp_scaled]]))[0][0]
#     temp_label = "Cold" if predicted_temp < 15 else "Warm" if predicted_temp < 25 else "Hot"
#     print(f"Predicted Temperature: {predicted_temp:.2f}°C ({temp_label})")

# Example usage
def predict_temperature(outlook, humidity, windy, play):
    input_data = pd.DataFrame([[outlook, humidity, windy, play]], columns=["Outlook", "Humidity", "Windy", "Play"])
    
    for column in input_data.columns:
        if column in label_encoders:
            if input_data[column][0] in label_encoders[column].classes_:
                input_data[column] = label_encoders[column].transform([input_data[column][0]])[0]
            else:
                input_data[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])[0]  # Assign default known class
    
    predicted_temp_scaled = model.predict(input_data.reshape(1, -1))[0]  # Ensure correct shape
    predicted_temp = scaler.inverse_transform([[predicted_temp_scaled]])[0][0]  # Correct 2D input
    temp_label = "Cold" if predicted_temp < 15 else "Warm" if predicted_temp < 25 else "Hot"
    print(f"Predicted Temperature: {predicted_temp:.2f}°C ({temp_label})")

# Test again
predict_temperature("Sunny", "High", "False", "Yes")

predict_temperature("Sunny", "High", "False", "Yes")
predict_temperature("Overcast", "Normal", "True", "No")