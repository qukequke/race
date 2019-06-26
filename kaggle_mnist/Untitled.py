import pandas as pd
import numpy as np
# import torch
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn import model_selection


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(train_df.head())
print(train_df.loc[:2, :])
print(train_df.iloc[:2, :])
train_np = train_df.iloc[:, :].get_values() / 255
train_np[:, 0] = train_np[:, 0] * 255


def get_weight(shape):
    w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return w
def get_bias(shape):
    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b

input_ = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='inputs_')
labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
w1 = get_weight([5,5, 1,32])
b1 = get_bias([32])
w2 = get_weight([5,5,32,64])
b2 = get_bias([64])
conv1 = tf.nn.conv2d(input_, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
conv1_relu = tf.nn.relu(conv1)
conv1_relu = tf.nn.max_pool(conv1_relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

conv2 = tf.nn.conv2d(conv1_relu, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
conv2_relu = tf.nn.relu(conv2)
conv2_relu = tf.nn.max_pool(conv2_relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

print(conv2_relu)
fc1_ = tf.reshape(conv2_relu, [-1, 7 * 7 *64])
print(fc1_)
fc1 = tf.layers.dense(fc1_, 1024, activation=tf.nn.relu)
logits = tf.layers.dense(fc1, 10, activation=None)
final = tf.nn.softmax(logits)
print(labels)
print(input_)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

[train_split, test_split] = model_selection.train_test_split(train_np, train_size=0.7, test_size=0.3)

def get_batch(train_split, batch_size):
    batches = []
    for i in range(train_split.shape[0]):
        if i % batch_size == 0:
            batch_x = train_split[i:i+batch_size, 1:]
            batch_y = train_split[i:i+batch_size, 0]
            # print(batch_y)
#             print(batch_y)
#             with tf.Session() as sess:
#                 batch_y = sess.run(tf.one_hot(batch_y, 10))
            batch_y = convert_to_one_hot(np.asarray(batch_y, np.int), 10)
            # print(batch_y)
            batches.append([batch_x, batch_y])
    return batches

def convert_to_one_hot(y, C):
    y = np.asarray(y, np.int)
    return np.eye(C)[y.reshape(-1)]

all_data = train_np[:, 1:].reshape(-1, 28, 28, 1)
all_label_one = convert_to_one_hot(train_np[:, 0], 10)
print(all_label_one.shape)
print(train_np[:, 1:].reshape(-1, 28, 28, 1).shape)
batches = get_batch(train_split, 200)
# test_batches = get_batch(test_split, 200)
# for x, y in batches:
#     print(x.shape)
#     print(y.shape)
# for i in range(100):
#     plt.imshow(all_data[i, :, :, 0])
#     plt.title(all_label_one[i])
#     plt.show()

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1)), dtype=tf.float32))
epochs = 10
with tf.Session() as sess:
    # plt.ion()
    sess.run(tf.global_variables_initializer())
    losses = []
    i = 1
    batch_nums = len(batches)
    # fig, ax = plt.subplots(1, 3)
    acces = []
    for epoch in range(epochs):
        for batch_x, batch_y in batches:
            batch_x = np.asarray(batch_x, np.float32)
            batch_x = batch_x.reshape(-1, 28, 28, 1)
            _, loss_, logits_= sess.run([optimizer, loss, logits], feed_dict={input_:batch_x, labels:batch_y})
            losses.append(loss_)

            # ax[0].plot(range(i), losses)
            # plt.ylim(1, 3)
            acc = sess.run(accuracy, feed_dict={input_: all_data[:2000], labels: all_label_one[:2000]})
            final_ = sess.run(final, feed_dict={input_: batch_x[0].reshape(-1, 28, 28, 1), labels: batch_y[0:1]})
            print(batch_x[0, :, :, 0].shape)
            # ax[2].imshow(batch_x[0, :, :, 0])
            # ax[2].set_title(np.argmax(final_))
            acces.append(acc)
            print(acc)
            # ax[1].plot(range(i), acces)
            # plt.pause(2)
            i += 1
            # plt.show()
            print('loss = ' + str(loss_))
            print('batch_nums is ' + str(batch_nums) + '  This is ' + str(i) + 'th')
            # if i % 50 == 0:
                # acc = sess.run(accuracy,feed_dict={input_: train_np[:200, 1:].reshape(-1, 28, 28, 1), labels: all_label_one[:200, :]})
                # acc = sess.run(accuracy,feed_dict={input_: all_data[:2000], labels: all_label_one[:2000]})
                # print('acc = ' + str(acc))
    plt.savefig('1.png')

    plt.ioff()

