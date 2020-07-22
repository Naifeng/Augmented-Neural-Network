import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

FILENAME = "max_pooling_gpu_500_points_Tesla_2.csv"



data = np.array(pd.read_csv(FILENAME))

train_data = data[:250]
test_data = data[250:]


train_feature = np.array(train_data[:, [6]])


train_label = np.array(train_data[:, [0]])


test_x = np.array(test_data[:, [6]])




print(test_data.shape)


x = tf.placeholder(tf.float32, [None, 1])  
y = tf.placeholder(tf.float32, [None, 1])  
train_feature = preprocessing.scale(train_feature) 
test_xs = preprocessing.scale(test_x)  
print(test_xs.shape)


w = tf.Variable(np.random.normal(), name='W')
b = tf.Variable(np.random.normal(), name='b')
prediction = tf.add(tf.multiply(w,x), b)


loss = tf.reduce_mean(tf.square(y - prediction))
saver = tf.train.Saver()


train_step = tf.train.AdamOptimizer(0.01).minimize(loss)


total_parameters = 0
for variable in tf.trainable_variables():

    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print("total parameters: ", total_parameters)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print(sess.run(loss, feed_dict={x: train_feature, y: train_label}))

    for i in range(5000):
        sess.run(train_step, feed_dict={x: train_feature, y: train_label})
        if i % 200 == 0:
            print(i)

            current_loss, current_A, current_b = sess.run([loss, w, b], feed_dict={
                x: train_feature,
                y: train_label
            })
            print(i, current_loss, current_A, current_b)


    prd = sess.run(prediction, feed_dict={x: test_xs}) 
    f = open('re_gpu.txt', 'w')

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

    saver.save(sess, "model/my-model")
