import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ... (rest of the code as in the previous response)

# Combine predictions with actual values in a new DataFrame
df_combined = pd.DataFrame({
    "Date": df["Date"],  # Assuming you have a "Date" column in your CSV
    "Open": df["Open"],
    "Volume": df["Volume"],
    "Actual Close": df[target_column],
    "Predicted Close": y_pred
})

# Set the index to the "Date" column
df_combined.set_index("Date", inplace=True)

# Plot the actual and predicted Close prices
plt.plot(df_combined["Actual Close"], label="Actual Close")
plt.plot(df_combined["Predicted Close"], label="Predicted Close")
plt.title("Actual vs. Predicted Close Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
