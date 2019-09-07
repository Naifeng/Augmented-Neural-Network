import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

FILENAME = "matrix_conv_cpu_500_points_I5.csv"


data = np.array(pd.read_csv(FILENAME))

train_data = data[:250] 
test_data = data[250:]

train_feature = np.array(train_data[:, [1, 2, 3, 4, 5]]) 

train_label = np.array(train_data[:, [0]])

test_x = np.array(test_data[:, [1, 2, 3, 4, 5]])


print(test_data.shape)


x = tf.placeholder(tf.float32, [None, 5])  
y = tf.placeholder(tf.float32, [None, 1])  
train_feature = preprocessing.scale(train_feature)  
test_xs = preprocessing.scale(test_x)  
print(test_xs.shape)


w = tf.Variable(tf.zeros([1, 5]), name='W')
b = tf.Variable(tf.zeros([]), name='b')
prediction = tf.add(tf.multiply(w,x), b)


loss = tf.reduce_mean(tf.square(y - prediction))
saver = tf.train.Saver()

train_step = tf.train.AdamOptimizer(0.001).minimize(loss) 


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

    for i in range(10000): 
        sess.run(train_step, feed_dict={x: train_feature, y: train_label})
        if i % 200 == 0:

            current_loss, current_A, current_b = sess.run([loss, w, b], feed_dict={
                x: train_feature,
                y: train_label
            })
            print(i, current_loss, current_A, current_b)

    prd = sess.run(prediction, feed_dict={x: test_xs}) 
    f = open('re.txt', 'w')

    for i in range(test_data.shape[0]):
        f.writelines(str(prd[i][0]) + "\n")
    f.close()


    saver.save(sess, "model/my-model")
