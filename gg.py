import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv("all_stocks_5yr.csv")

# Convert the "Date" column to datetime format
data['date'] = pd.to_datetime(data['date'])


sns.lineplot(data=data, x='high', y='close')
plt.show()
