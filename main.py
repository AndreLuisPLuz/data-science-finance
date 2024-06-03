import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from scipy.stats import f_oneway

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

    # Teste ANOVA para os retornos mensais. Esse teste aceita ou rejeita a hipótese nula:
    # de que a média dos retornos para o mês, irrespectivo do ano, é igual para todos os
    # meses.
    monthly_returns = [data[data['Month'] == month]['Return'].dropna() for month in range(1, 13)]
    anova_result = f_oneway(*monthly_returns)
    print('ANOVA result for monthly returns:', anova_result)

    # Mesmo teste, para os retornos semestrais.
    quarterly_returns = [data[data['Quarter'] == quarter]['Return'].dropna() for quarter in data['Quarter'].unique()]
    anova_result = f_oneway(*quarterly_returns)
    print('ANOVA result for quarterly returns:', anova_result)