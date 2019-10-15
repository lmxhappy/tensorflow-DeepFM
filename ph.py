#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/9/30
'''
import tensorflow as tf
import numpy as np

x1 = tf.placeholder(tf.float32, shape=(2, 2))
x2 = tf.placeholder(tf.float32, shape=(2))
# y = tf.matmul(x1, x2)
y1 = tf.multiply(x1, x2)

with tf.Session() as sess:
    # print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array1 = np.random.rand(2, 2)
    rand_array2 = np.random.rand(2)
    print(sess.run([y1], feed_dict={x1: rand_array1, x2:rand_array2}))  # Will succeed.