# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset
# Assuming you have a CSV file named 'data.csv' with columns: Open, Volume, Close
data = pd.read_csv('spy.csv')

# Split the dataset into features (X) and target variable (y)
X = data[['Open', 'Volume']]
y = data['Close']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate XGBoost regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
                           colsample_bytree = 0.3,
                           learning_rate = 0.1,
                           max_depth = 5,
                           alpha = 10,
                           n_estimators = 10)

# Fit the regressor to the training set
xg_reg.fit(X_train,y_train)

# Predict on the test set
y_pred = xg_reg.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
from sklearn.metrics import r2_score

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)

import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Close')
plt.ylabel('Predicted Close')
plt.title('Actual vs Predicted Close')
plt.legend()
plt.show()
import matplotlib.pyplot as plt

# Plot actual Close values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, color='blue', label='Actual Close')
plt.xlabel('Index')
plt.ylabel('Close Value')
plt.title('Actual Close Values')
plt.legend()
plt.show()

data['Close'].plot()