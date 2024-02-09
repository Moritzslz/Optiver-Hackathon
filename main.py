import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a DataFrame
data = pd.read_csv('fsa_case_train.csv')

# Assuming that the first column of the CSV contains the date, set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Specify the columns you want to plot
columns_to_plot = ['CONS_CONF', 'MACRO_VOL', 'OIL', 'YIELDS', 'ELECTRICITY', 'FORD_VOL', 'BTC_VOL', 'MUSK_POP', 'CURRENCY_VOL', 'TSLA_VOL']
columns_to_plot_2 = ['CONS_CONF', 'MACRO_VOL', 'FORD_VOL', 'BTC_VOL', 'MUSK_POP', 'CURRENCY_VOL', 'TSLA_VOL']
CONS_CONF = ['CONS_CONF', 'TSLA_VOL']
MACRO_VOL = ['MACRO_VOL', 'TSLA_VOL']
FORD_VOL = ['FORD_VOL', 'TSLA_VOL']
BTC_VOL = ['BTC_VOL', 'TSLA_VOL']
MUSK_POP = ['MUSK_POP', 'TSLA_VOL']
CURRENCY_VOL = ['CURRENCY_VOL', 'TSLA_VOL']


# Plot the data
data[CONS_CONF].plot(figsize=(12, 6))
data[MACRO_VOL].plot(figsize=(12, 6))
data[FORD_VOL].plot(figsize=(12, 6))
data[BTC_VOL].plot(figsize=(12, 6))
data[MUSK_POP].plot(figsize=(12, 6))
data[CURRENCY_VOL].plot(figsize=(12, 6))
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend(title='Columns', loc='upper left')
plt.grid(True)
plt.show()
