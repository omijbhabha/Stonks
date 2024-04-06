import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf

# Load historical stock data
data = yf.download("TCS.BO",period="3y",actions=True)  # Replace "historical_stock_data.csv" with your dataset file

# Feature engineering: Extract relevant features from the data
# Example features could include moving averages, RSI, MACD, etc.
# For simplicity, let's assume 'Open', 'High', 'Low', 'Close', 'Volume' are our features
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Define the target variable: Whether the stock price increased or decreased
data['Price_Increase'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Prepare feature matrix and target variable
X = data[features]
y = data['Price_Increase']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Now, you can use this trained model to predict future stock trends
# You would need to have the same set of features for the future data and pass it to the trained model for prediction
# Remember, this is a simplified example and may require additional preprocessing and feature engineering for better performance
