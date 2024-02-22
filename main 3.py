import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def predict_close(data_percentage=0.2):
    # Load the dataset
    df = pd.read_csv("spy.csv")

    # Selecting features and target variable
    X = df[['Open', 'Volume']]
    y = df['Close']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_percentage, random_state=42)

    # Creating XGBoost Regression model
    model = XGBRegressor()

    # Training the model
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Calculating Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Mean Absolute Percentage Error:", mape)

    # Calculating R^2 score
    r2 = r2_score(y_test, y_pred)
    print("R^2 Score:", r2)

    # Plotting the data and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Close')
    plt.plot(y_pred, label='Predicted Close', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Price')
    plt.legend()
    plt.show()

# Example usage with 30% of the data used for prediction
predict_close(data_percentage=0.1)
