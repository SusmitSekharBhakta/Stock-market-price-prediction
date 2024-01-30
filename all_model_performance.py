from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import FileLink
import csv
import warnings
# Ignore warnings
warnings.filterwarnings("ignore")
# Example evaluation metrics data
models = ['Simple LSTM', 'Bi-LSTM', 'Simple GRU', 'Bi-GRU', 'GRU-LSTM', 'LSTM-GRU-Bi-LSTM', 'Stacked LSTM', 'CNN-Bi-LSTM', 'EEMD-Ensemble CNN', 'EEMD-Ensemble-CNN(Large)']
MSE = [0.0004072232445763366, 0.0005000506139239725, 0.00022834628914830254, 0.0012235577912200863, 0.0009060262745021321, 0.002075882273433005, 0.0009451684151423362, 0.0009223105824260224, 0.00019808381330221891, 0.0001667082]
MAE = [0.016059828137190243, 0.018079012228853664, 0.011964837421536265, 0.030203257378998514, 0.02426845818742699, 0.03896816927287234, 0.025767192679360194, 0.025158315670018652, 0.009464983828365803, 0.0092958861]
sMAPE = [0.018891302600025413, 0.021166436757949537, 0.01407366311680603, 0.03577723942453357, 0.02837164567556185, 0.045901362491680725, 0.030379938861843116, 0.029672016395640002, 0.008302686735987663, 0.0080958734]
R2_score = [0.8910212191905597, 0.8521526418563456, 0.9403095873293223, 0.6696787248501639, 0.6285710529142268, 0.2031777307201621, 0.7169999934312372, 0.7258388456358698, 0.9876543208956718, 0.9900237588]

# Combine the data into a list of lists
data = [models, MSE, MAE, sMAPE, R2_score]

# Transpose the data so that each model's metrics are in a row
data = list(map(list, zip(*data)))

# Print the table using tabulate
headers = ['Model', 'MSE', 'MAE', 'sMAPE', 'R2-Score']
print(tabulate(data, headers=headers, tablefmt='orgtbl'))

# Write table to CSV file
with open('metrics_table.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'MSE', 'MAE', 'sMAPE', 'R2-Score'])
    writer.writerows(data)

# Make table downloadable
display(FileLink('updated_metrics_table.csv'))

# Convert the metric scores to float and round off to two decimal places
MSE = [round(score, 5) for score in MSE]
MAE = [round(score, 4) for score in MAE]
sMAPE = [round(score, 5) for score in sMAPE]
R2_score = [round(score, 5) for score in R2_score]

# Create a figure with subplots for each metric
fig, axs = plt.subplots(2, 2, figsize=(12, 16))

# Plot the MSE for each model
axs[0, 0].bar(models, MSE, color='cyan')
axs[0, 0].set_xticklabels(models, rotation=45, ha='right')
axs[0, 0].set_ylabel('MSE')
axs[0, 0].set_title('MSE for Different Models')

# Plot the MAE for each model
axs[0, 1].bar(models, MAE, color='green')
axs[0, 1].set_xticklabels(models, rotation=45, ha='right')
axs[0, 1].set_ylabel('MAE')
axs[0, 1].set_title('MAE for Different Models')

# Plot the sMAPE for each model
axs[1, 0].bar(models, sMAPE, color='orange')
axs[1, 0].set_xticklabels(models, rotation=45, ha='right')
axs[1, 0].set_ylabel('sMAPE')
axs[1, 0].set_title('sMAPE for Different Models')

# Plot the R2-Score for each model
axs[1, 1].bar(models, R2_score, color='red')
axs[1, 1].set_xticklabels(models, rotation=45, ha='right')
axs[1, 1].set_ylabel('R2-Score')
axs[1, 1].set_title('R2-Score for Different Models')

# Adjust the layout and display the figure
fig.tight_layout()
plt.show()
