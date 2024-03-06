# Importing necessary libraries


import datetime
import json
import sklearn

import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

tf.random.set_seed(
    99
)

"""# Data loading and pre processing"""

dataFrame = pd.read_csv('/content/final_data_adj.csv')
dataFrame.head()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan)
dataFrame.drop(columns=['Date'], inplace=True)
dataFrame = pd.DataFrame(imputer.fit_transform(dataFrame), columns=dataFrame.columns)
dataFrame = dataFrame.reset_index(drop=True)

dataFrame.isna().sum()

dataFrame.head()

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(dataFrame.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(dataFrame.columns))

target_scaler = MinMaxScaler(feature_range=(0, 1))

df_scaled.Close = target_scaler.fit_transform(dataFrame.Close.to_numpy().reshape(-1, 1))

df_scaled=df_scaled.astype(float)

from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=16, gamma=3, kernel='poly')

kpca.fit(df_scaled)

kpca.transform(df_scaled)

def singleStepSampler(df, window):
    xRes = []
    yRes = []
    for i in range(0, len(df)-window):
        res = []
        for j in range(0, window):
            r = []
            for col in df.columns:
                r.append(df[col][i+j])
            res.append(r)
        xRes.append(res)
        yRes.append(df.Close[i+window])
    return (np.array(xRes), np.array(yRes))

SPLIT = 0.85
(xVal, yVal) = singleStepSampler(df_scaled, 20)
X_train = xVal[:int(SPLIT*len(xVal))]
y_train = yVal[:int(SPLIT*len(yVal))]
X_test = xVal[int(SPLIT*len(xVal)):]
y_test = yVal[int(SPLIT*len(yVal)):]
(X_train.shape, X_test.shape)

"""## Custom metrics for evaluation"""

K = tf.keras.backend

def sMAPE(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape

"""# Simple LSTM"""

simple_lstm = tf.keras.Sequential()
simple_lstm.add(tf.keras.layers.LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
simple_lstm.add(tf.keras.layers.Dropout(0.2))
simple_lstm.add(tf.keras.layers.Dense(1, activation='linear'))
simple_lstm.compile(loss = 'MeanSquaredError', metrics=['MAE', 'MSE', 'MAPE', sMAPE], optimizer='Adam')
simple_lstm.fit(X_train, y_train, epochs=20)

tf.keras.utils.plot_model(simple_lstm)

"""# BI-LSTM"""

bi_lstm = tf.keras.Sequential()
bi_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
bi_lstm.add(tf.keras.layers.Dropout(0.1))
bi_lstm.add(tf.keras.layers.Flatten())
bi_lstm.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

bi_lstm.compile(loss = 'MeanSquaredError', metrics=['MAE', 'MSE', 'MAPE', sMAPE], optimizer=tf.keras.optimizers.Adam())
bi_lstm.fit(X_train, y_train, epochs=20)

tf.keras.utils.plot_model(bi_lstm)

"""# **Simple_GRU**"""

simple_gru = tf.keras.Sequential()
simple_gru.add(tf.keras.layers.GRU(200, input_shape=(X_train.shape[1], X_train.shape[2])))
simple_gru.add(tf.keras.layers.Dropout(0.5))
simple_gru.add(tf.keras.layers.Dense(1, activation='linear'))
simple_gru.compile(loss = 'MeanSquaredError', metrics=['MAE', 'MSE', 'MAPE', sMAPE], optimizer=tf.keras.optimizers.Adam())
simple_gru.fit(X_train, y_train, epochs=20)

tf.keras.utils.plot_model(simple_gru)

"""# BI-GRU"""

bi_gru = tf.keras.Sequential()
bi_gru.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
bi_gru.add(tf.keras.layers.Dropout(0.1))
bi_gru.add(tf.keras.layers.Flatten())
bi_gru.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

bi_gru.compile(loss = 'MeanSquaredError', metrics=['MSE','MAPE','MAE', sMAPE], optimizer=tf.keras.optimizers.Adam())
bi_gru.fit(X_train, y_train, epochs=20)

tf.keras.utils.plot_model(bi_gru)

gru_lstm = tf.keras.Sequential()
gru_lstm.add(tf.keras.layers.GRU(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
gru_lstm.add(tf.keras.layers.Dropout(0.4))
gru_lstm.add(tf.keras.layers.GRU(128, return_sequences=True))
gru_lstm.add(tf.keras.layers.Dropout(0.4))
gru_lstm.add(tf.keras.layers.LSTM(64, return_sequences=True))
gru_lstm.add(tf.keras.layers.LSTM(64))
gru_lstm.add(tf.keras.layers.Dropout(0.2))
gru_lstm.add(tf.keras.layers.Dense(1, activation='linear'))

gru_lstm.compile(loss = 'MeanSquaredError', metrics=['MSE','MAPE','MAE', sMAPE], optimizer=tf.keras.optimizers.Adam())
gru_lstm.fit(X_train, y_train, epochs=20)

lstm_gru_bi_lstm = tf.keras.Sequential()
lstm_gru_bi_lstm.add(tf.keras.layers.GRU(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
lstm_gru_bi_lstm.add(tf.keras.layers.Dropout(0.4))
lstm_gru_bi_lstm.add(tf.keras.layers.LSTM(128, return_sequences=True))
lstm_gru_bi_lstm.add(tf.keras.layers.Dropout(0.4))
lstm_gru_bi_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
lstm_gru_bi_lstm.add(tf.keras.layers.LSTM(64))
lstm_gru_bi_lstm.add(tf.keras.layers.Dropout(0.2))
lstm_gru_bi_lstm.add(tf.keras.layers.Dense(1, activation='linear'))

lstm_gru_bi_lstm.compile(loss = 'MeanSquaredError', metrics=['MSE','MAPE','MAE', sMAPE], optimizer=tf.keras.optimizers.Adam())
lstm_gru_bi_lstm.fit(X_train, y_train, epochs=20)

stacked_lstm=tf.keras.Sequential()
stacked_lstm.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
stacked_lstm.add(tf.keras.layers.LSTM(50, return_sequences=True))
stacked_lstm.add(tf.keras.layers.LSTM(50))
stacked_lstm.add(tf.keras.layers.Dense(1, activation='linear'))

stacked_lstm.compile(loss = 'MeanSquaredError', metrics=['MSE','MAPE','MAE', sMAPE], optimizer=tf.keras.optimizers.Adam())
stacked_lstm.fit(X_train, y_train, epochs=20)

cnn_bi_lstm = tf.keras.Sequential()
cnn_bi_lstm.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
cnn_bi_lstm.add(tf.keras.layers.MaxPooling1D(pool_size=2))
cnn_bi_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
cnn_bi_lstm.add(tf.keras.layers.Dense(1, activation='linear'))

cnn_bi_lstm.compile(loss = 'MeanSquaredError', metrics=['MSE','MAPE','MAE', sMAPE], optimizer=tf.keras.optimizers.Adam())
cnn_bi_lstm.fit(X_train, y_train, epochs=20)

lstm_cnn_gru = tf.keras.Sequential()
lstm_cnn_gru.add(tf.keras.layers.Conv1D(32, 4, activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_cnn_gru.add(tf.keras.layers.LSTM(50, return_sequences=True))
lstm_cnn_gru.add(tf.keras.layers.Dropout(0.4))
lstm_cnn_gru.add(tf.keras.layers.MaxPooling1D(2))
lstm_cnn_gru.add(tf.keras.layers.Conv1D(32, 8, activation="relu", padding='same'))
lstm_cnn_gru.add(tf.keras.layers.LSTM(50, return_sequences=True))
lstm_cnn_gru.add(tf.keras.layers.Dropout(0.4))
lstm_cnn_gru.add(tf.keras.layers.MaxPooling1D(2))
lstm_cnn_gru.add(tf.keras.layers.Conv1D(64, 8, activation="relu", padding='same'))
lstm_cnn_gru.add(tf.keras.layers.LSTM(50))
lstm_cnn_gru.add(tf.keras.layers.Dropout(0.2))
lstm_cnn_gru.add(tf.keras.layers.Dense(1, activation='linear'))

lstm_cnn_gru.compile(loss = 'MeanSquaredError', metrics=['MSE','MAPE','MAE', sMAPE], optimizer=tf.keras.optimizers.Adam())
lstm_cnn_gru.fit(X_train, y_train, epochs=20)

d = {
     'Simple LSTM': simple_lstm.predict([X_test]).reshape(-1),
     'Bi-LSTM': bi_lstm.predict([X_test]).reshape(-1),
     'Simple GRU': simple_gru.predict([X_test]).reshape(-1),
     'Bi-GRU': bi_gru.predict([X_test]).reshape(-1),
     'GRU-LSTM': gru_lstm.predict([X_test]).reshape(-1),
     'LSTM-GRU-Bi-LSTM': lstm_gru_bi_lstm.predict([X_test]).reshape(-1),
     'Stacked LSTM': stacked_lstm.predict([X_test]).reshape(-1),
     'CNN-Bi-LSTM': cnn_bi_lstm.predict([X_test]).reshape(-1),
     'LSTM-CNN-GRU': lstm_cnn_gru.predict([X_test]).reshape(-1),
     'True': y_test.reshape(-1, 1).reshape(-1),
    }

d = pd.DataFrame(d)

fig, ax = plt.subplots(figsize=(18, 9))
plt.plot(d)
ax.legend(d.keys());
plt.show()

def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + 1e-10))

def eval(model):
  return {
      'MSE': sklearn.metrics.mean_squared_error(d[model].to_numpy(), d['True'].to_numpy()),
      'MAE': sklearn.metrics.mean_absolute_error(d[model].to_numpy(), d['True'].to_numpy()),
      'RMSE': math.sqrt(sklearn.metrics.mean_absolute_error(d[model].to_numpy(), d['True'].to_numpy())),
      'sMAPE': smape(d[model].to_numpy(), d['True'].to_numpy()),
      'R2': sklearn.metrics.r2_score(d[model].to_numpy(), d['True'].to_numpy())
  }

result = dict()

for item in d.keys():
  if item != 'True':
    result[item] = eval(item)

! cd '/content/'
! touch data-5-SS-kpca.json

fp = open('/content/data-5-SS-kpca.json', 'w')

json.dump(result, fp, indent=6)

fp.close()

result

import csv
from tabulate import tabulate
data = {
    'Simple LSTM': {
        'MSE': 0.0004072232445763366,
        'MAE': 0.016059828137190243,
        'RMSE': 0.1267273772205132,
        'sMAPE': 0.018891302600025413,
        'R2': 0.8910212191905597
    },
    'Bi-LSTM': {
        'MSE': 0.0005000506139239725,
        'MAE': 0.018079012228853664,
        'RMSE': 0.13445821740917757,
        'sMAPE': 0.021166436757949537,
        'R2': 0.8521526418563456
    },
    'Simple GRU': {
        'MSE': 0.00022834628914830254,
        'MAE': 0.011964837421536265,
        'RMSE': 0.10938389927926442,
        'sMAPE': 0.01407366311680603,
        'R2': 0.9403095873293223
    },
    'Bi-GRU': {
        'MSE': 0.0002235577912200863,
        'MAE': 0.010203257378998514,
        'RMSE': 0.1037908437720426,
        'sMAPE': 0.00977723942453357,
        'R2': 0.9696787248501639
    },
    'GRU-LSTM': {
        'MSE': 0.0009060262745021321,
        'MAE': 0.02426845818742699,
        'RMSE': 0.15578336941864812,
        'sMAPE': 0.02837164567556185,
        'R2': 0.6285710529142268
    },
    'LSTM-GRU-Bi-LSTM': {
        'MSE': 0.002075882273433005,
        'MAE': 0.03896816927287234,
        'RMSE': 0.1974035695545355,
        'sMAPE': 0.045901362491680725,
        'R2': 0.2031777307201621
    },
    'Stacked LSTM': {
        'MSE': 0.0009451684151423362,
        'MAE': 0.025767192679360194,
        'RMSE': 0.16052162682754056,
        'sMAPE': 0.030379938861843116,
        'R2': 0.7169999934312372
    },
    'CNN-Bi-LSTM': {
        'MSE': 0.0009223105824260224,
        'MAE': 0.025158315670018652,
        'RMSE': 0.15861373102609577,
        'sMAPE': 0.029672016395640002,
        'R2': 0.7258388456358698
    },
    'LSTM-CNN-GRU': {
        'MSE': 0.04377254406134014,
        'MAE': 0.20543745768350757,
        'RMSE': 0.4532520906554183,
        'sMAPE': 0.2708600441748303,
        'R2': -29.092252035841888
    }
}

header = ['Model', 'MSE', 'MAE', 'RMSE', 'sMAPE', 'R2']
rows = []

# Convert the data into a list of rows
for model, metrics in data.items():
    row = [model] + [metrics[key] for key in header[1:]]
    rows.append(row)

# Save the data as a CSV file
filename = 'table_data.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

# Display the table in the RunShell
table = [header] + rows
print(tabulate(table, headers='firstrow'))
