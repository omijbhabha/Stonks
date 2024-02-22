import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def predict_close(data_percentage=0.2):
    # Load the dataset
    data = pd.read_csv("spy.csv")

    # Selecting features and target variable
    features= [['Open', 'Volume']]
    target = ['Close']

    # Splitting the dataset into training and testing sets
    train = data.iloc[:int(.10 * len(data)), :]
    test = data.iloc[int(.10 * len(data)):, :]
    # Creating XGBoost Regression model
    model = XGBRegressor()

    # Training the model
    model.fit(train, test)

    # Predicting on the test set
    prediction = model.predict(test[features])
    print("Model Predictions")
    print(prediction)

    print("Actual values")
    print(test[target])

    # Plotting the data and predicted values
    plt.plot(data['Close'], label='Close Price')
    plt.plot(test[target].index, prediction, label='Predictions')
    plt.legend()
    plt.show()

# Example usage with 30% of the data used for prediction
predict_close(data_percentage=0.1)
