import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical data for TCS stock
tcs = yf.Ticker("TCS.BO")  # TCS.BO is the Yahoo Finance ticker symbol for TCS on the Bombay Stock Exchange (BSE)
hist_data = tcs.history(period="10y")

# Plotting the closing price over the last 10 years
plt.figure(figsize=(10, 6))
plt.plot(hist_data.index, hist_data['Close'], label='TCS Stock Price')
plt.title('TCS Stock Price Over the Last 10 Years')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.legend()
plt.grid(True)
plt.show()
