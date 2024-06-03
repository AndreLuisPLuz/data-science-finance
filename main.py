import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv('stocks_prediction.csv')

    data['Date'] = pd.to_datetime(data['Date'])

    data = data.sort_values('Date')

    data['Return'] = data['Close'].pct_change()

    plt.figure(figsize=(12, 6))
    for stock in data['Stock Name'].unique():
        stock_data = data[data['Stock Name'] == stock]
        plt.plot(stock_data['Date'], stock_data['Close'], label=stock)

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Prices Over Time')
    plt.legend()
    plt.show()