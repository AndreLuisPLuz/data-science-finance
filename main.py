import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

if __name__ == "__main__":
    # Organizando os dados por data
    data = pd.read_csv('stocks_prediction.csv')

    data['Date'] = pd.to_datetime(data['Date'])

    data = data.sort_values('Date')

    # Estimando os ganhos diários
    data['Return'] = data['Close'].pct_change()

    # Plotando a série histórica completa
    plt.figure(figsize=(12, 6))
    for stock in data['Stock Name'].unique():
        stock_data = data[data['Stock Name'] == stock]
        plt.plot(stock_data['Date'], stock_data['Close'], label=stock)

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Prices Over Time')
    plt.legend()
    plt.show()

    # Decompondo a série história de acordo com uma ação especícia com STL
    stock_data = data[data['Stock Name'] == 'PETR4.SA']
    stock_data = stock_data.set_index('Date')

    stl = STL(stock_data['Close'], period=365, seasonal=731)
    result = stl.fit()

    result.plot()
    plt.show()

    # Conseguindo os retornos mensais
    data['Month'] = data['Date'].dt.month
    monthly_returns = data.groupby(['Month', 'Stock Name'])['Return'].mean().unstack()

    monthly_returns.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Average Return')
    plt.show()

    # Conseguindo os retornos trimestrais
    data['Quarter'] = data['Date'].dt.quarter
    quarterly_returns = data.groupby(['Quarter', 'Stock Name'])['Return'].mean().unstack()

    quarterly_returns.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Quarterly Returns')
    plt.xlabel('Quarter')
    plt.ylabel('Average Return')
    plt.show()

