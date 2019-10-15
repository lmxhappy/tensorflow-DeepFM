#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/9/30
'''
#!/usr/bin/env/python
# coding=utf-8
import tensorflow as tf
import numpy as np

input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

embedding = tf.Variable(np.zeros((259, 8), dtype=np.int32))
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# print(embedding.eval())
input_ids_list = np.random.rand(1024, 39)
print('------------')
# print(sess.run((input_embedding, embedding), feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))
input_embedding_p, embedding_p = sess.run((input_embedding, embedding), feed_dict={input_ids:list(input_ids_list)})
print(input_embedding_p)
print(input_embedding_p.shape)
print(embedding_p)
