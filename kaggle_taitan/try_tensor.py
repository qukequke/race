import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def split_data(df):
    split_train, split_test = train_test_split(df)
    return split_train, split_test


train_df = pd.read_csv('feature.csv')
test_df = pd.read_csv('test.csv')

train_df_array = np.array(train_df, dtype='float32')
test_df_array = np.array(test_df, dtype='float32')

split_train, split_test = split_data(train_df_array)
print(split_train.shape)

feature_num = len(train_df.columns)-1
Node = 100
Node2 = 100
iterations = 40000

train_row, train_column = train_df.as_matrix().shape

x = tf.placeholder(dtype='float', shape=[None, train_column-1])
y = tf.placeholder(dtype='float', shape=[None, 1])

w1 = tf.get_variable(name='w1', shape=[feature_num, Node], initializer=tf.contrib.layers.xavier_initializer(seed=0))
b1 = tf.get_variable(name='b1', shape=[Node], initializer=tf.zeros_initializer)

w2 = tf.get_variable(name='w2', shape=[Node, Node2], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name='b2', shape=[Node2], initializer=tf.zeros_initializer)

w3 = tf.get_variable(name='w3', shape=[Node2, 1], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name='b3', shape=[1], initializer=tf.zeros_initializer)

l1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
l2 = tf.nn.relu(tf.add(tf.matmul(l1, w2), b2))
l3 = tf.add(tf.matmul(l2, w3), b3)
y_ = tf.nn.sigmoid(l2)

predict = (y_ > 0.5)
correct_prediction = tf.equal(predict, (y > 0.5))
accarcy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

reg = 0.01*(tf.reduce_mean(tf.square(w1)) + tf.reduce_mean(tf.square(w2)))

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.maximum(0.00001, y)) + (1.0 - y_)*tf.log(tf.maximum(0.00001, 1.0-y)))
# cross_entropy2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=l2)

loss = cross_entropy + reg

optimer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cross_entropy)
# optimer2 = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cross_entropy2)

feed_x = train_df_array[:, 1:]
feed_y = train_df_array[:, 0].reshape(-1, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        sess.run(optimer, feed_dict={x: feed_x, y: feed_y})
        if i % 1000 == 0:
            print('{} {} {:.2f}%'.format(i, sess.run(cross_entropy,feed_dict={x: feed_x, y: feed_y}),sess.run(accarcy, feed_dict={x: feed_x, y: feed_y})*100.0))
    print('---'*20)
    print('test score : {:.2f}%'.format(sess.run(accarcy, feed_dict={x: split_test[:, 1:], y: split_test[:, 0].reshape(-1, 1)})*100.0))
    pre_y_bool = sess.run(predict, feed_dict={x: test_df_array})
    pre_y = pd.Series(pre_y_bool.reshape(-1)).map({True: 1, False: 0})
    sub_pd = pd.DataFrame()
    aaaa = pd.read_csv('data/test.csv')
    sub_pd['PassengerId'] = aaaa.PassengerId
    sub_pd['Survived'] = pre_y
    # sub_pd.to_csv('my_tensorflow.csv', index=False)





