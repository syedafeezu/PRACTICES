
#LINBRARIES ON THE WAYYYYYY

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,  GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# LOADINGG>>>>>>
try:
    df = pd.read_csv("weather_forecast.csv")
except FileNotFoundError:
    raise FileNotFoundError("The dataset file 'weather_forecast.csv' was not found.")

# ENCODINGGG...
label_encoders = {}
categorical_columns = ["Outlook", "Humidity", "Windy", "Play", "Temperature"]

for column in categorical_columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# LETS GOOOO FOR THE TARGETTT
if "Temperature" not in df.columns:
    raise KeyError("The dataset does not contain a 'Temperature' column.")

X = df.drop(columns=["Temperature"])
y = df["Temperature"]

# Check for NaN values
df.fillna(df.mean(), inplace=True)

# Splitting to TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ACTUAL FUNN

# model = RandomForestRegressor(n_estimators=100, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# model = LinearRegression()
# model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# PREDICTION HAPPENS HERE.....!~~~~~
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

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

# Examplezzzz
predict_temperature("Sunny", "High", "False", "Yes")

predict_temperature("Overcast", "Normal", "True", "No")

predict_temperature("Rainy", "High", "True", "No")

predict_temperature("Sunny", "Normal", "False", "Yes")