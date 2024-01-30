!pip install EMD-signal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def eemd(x, num_sifts, num_modes):
    def sift(x):
        # Sifting process
        mean = np.mean(x)
        upper_env = []
        lower_env = []
        imf = np.zeros(len(x))

        for _ in range(num_sifts):
            upper = []
            lower = []
            h = x - mean

            for i in range(1, len(x) - 1):
                if (h[i] > 0 and h[i] > h[i - 1] and h[i] > h[i + 1]) or (
                        h[i] < 0 and h[i] < h[i - 1] and h[i] < h[i + 1]):
                    upper.append(i)
                elif (h[i] > 0 and h[i] < h[i - 1] and h[i] < h[i + 1]) or (
                        h[i] < 0 and h[i] > h[i - 1] and h[i] > h[i + 1]):
                    lower.append(i)

            if len(upper) < 3 or len(lower) < 3:
                break

            upper_env.extend(upper)
            lower_env.extend(lower)
            mean = np.mean(x[upper + lower])

        if len(upper_env) > 0 and len(lower_env) > 0:
            upper_env = np.array(upper_env)
            lower_env = np.array(lower_env)
            interp_upper = np.interp(range(len(x)), upper_env, x[upper_env])
            interp_lower = np.interp(range(len(x)), lower_env, x[lower_env])
            imf = (interp_upper + interp_lower) / 2

        return imf

    imfs = []
    residual = x.copy()

    for _ in range(num_modes):
        imf = sift(residual)
        imfs.append(imf)
        residual -= imf

    imfs.append(residual)

    return imfs


# Load the stock data from the CSV file
data = pd.read_csv("final_data_adj.csv")

# Perform data preprocessing
data['Date'] = pd.to_datetime(data['Date'])  # Convert the 'Date' column to datetime format
data.set_index('Date', inplace=True)  # Set the 'Date' column as the index
data = data[['RSI']]  # Select the 'RSI' column as the input for EEMD
# similarly select Closs and Sadj for corresponding IMF calculations
# Normalize the data
normalized_data = (data - data.mean()) / data.std()

# Extract the closing prices
closing_prices = normalized_data['RSI'].values
# similarly select Close and Sadj for corresponding IMF calculations
# Perform EEMD decomposition
num_sifts = 500
num_modes = 0

imfs = eemd(closing_prices, num_sifts, num_modes)

# Visualize the decomposed IMFs
num_imfs = len(imfs)

plt.figure(figsize=(10, 6))
for i in range(num_imfs):
    plt.subplot(num_imfs+1, 1, i+1)
    plt.plot(imfs[i], label=f'IMF {i+1}')
    plt.legend()
plt.subplot(num_imfs+1, 1, num_imfs+1)
plt.plot(imfs[-1], label='Residue')
plt.legend()
plt.show()
