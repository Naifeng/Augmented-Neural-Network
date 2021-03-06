import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

FILENAME = "matrix_conv_gpu_500_points_Tesla.csv"


data = np.array(pd.read_csv(FILENAME))

train_data = data[:250]
test_data = data[250:]

X_train = np.array(train_data[:, [1, 2, 3, 4, 5]])
y_train = np.array(train_data[:, [0]])

X_test = np.array(test_data[:, [1, 2, 3, 4, 5]])
Y_test = np.array(test_data[:, [1, 2, 3, 4, 5]])

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

reg = LinearRegression().fit(X_train, y_train)

prd = reg.predict(X_test)

f = open('re.txt', 'w')

for i in range(test_data.shape[0]):
    f.writelines(str(prd[i][0]) + "\n")
f.close()


# -----evaluation-----#

import math
import statistics

sum_ae = 0.0
sum_ape = 0.0
sum_aape = 0.0

truth_value_list = []

for i in range(test_data.shape[0]):
    truth_value = test_data[:, [0]][i][0]
    sum_ae += abs(prd[i][0] - test_data[:, [0]][i][0])
    truth_value_list.append(truth_value)

print("MAE: ", sum_ae / test_data.shape[0])


c = 0

# decide the percentage to drop
percentage = 0.3
threshold = sorted(truth_value_list)[int(len(test_data)*percentage) - 1]

median = statistics.median(truth_value_list)


for i in range(test_data.shape[0]):

    pred_value = prd[i][0]
    truth_value = test_data[:, [0]][i][0]

    ape = (abs(prd[i][0] - test_data[:, [0]][i][0]) / test_data[:, [0]][i][0])
    aape = math.atan(abs(prd[i][0] - test_data[:, [0]][i][0]) / test_data[:, [0]][i][0])

    # valid rule
    if truth_value > threshold:
        sum_ape += ape
        c += 1

    sum_aape += aape

print("MAPE: ", sum_ape / c)
print("MAAPE: ", sum_aape / test_data.shape[0])

print("threshold value:", threshold)
print("truth median:", median)
print("range from", min(truth_value_list), "to", max(truth_value_list))
print("valid points (MAPE):", c, "out of", test_data.shape[0])

# ------------------#
