import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


flags = tf.app.flags
FLAGS = flags.FLAGS


def mynet(input,keep_rate=0.7, reuse=None):
    with tf.name_scope("model"):
        print "input"
        print input.shape
        
        with tf.variable_scope("conv1") as scope:
            conv1 = tf.layers.conv3d(input, filters=16, kernel_size=[5,5,5], activation=tf.nn.relu, padding='SAME',
                reuse=reuse)
            #net = tf.layers.max_pooling3d(net,pool_size=[2, 2,2],strides=[1,1,1], padding='SAME')
            print "conv1"
            #print net.shape
        with tf.variable_scope("conv2") as scope:
            conv2 = tf.layers.conv3d(conv1, filters=64, kernel_size=[5,5,5], activation=tf.nn.relu, padding='SAME',
                reuse=reuse)
            pool1 = tf.layers.max_pooling3d(conv2,pool_size=[2, 2,2],strides=2, padding='SAME')
            print "conv2"
            #print net.shape
        with tf.variable_scope("conv3") as scope:
            conv3 = tf.layers.conv3d(pool1, 128, [5,5,5], activation=tf.nn.relu, padding='SAME',
                reuse=reuse)
            pool2 = tf.layers.max_pooling3d(conv3, pool_size=[2, 2,2],strides=[1,1,1], padding='SAME')
            print "conv3"
            #print net.shape
            
        with tf.variable_scope("conv4") as scope:
            conv4 = tf.layers.conv3d(pool2, 16, [5,5,5], activation=tf.nn.relu, padding='SAME',reuse=reuse)
            pool3 = tf.layers.max_pooling3d(conv4, pool_size=[2, 2,2],strides=2, padding='SAME')
            print "conv4"
            #print net.shape
           
        with tf.variable_scope("batch_norm") as scope:
            convbn = tf.layers.batch_normalization(inputs=pool3,training=True,reuse=reuse)
            print "batch norm"
            #print net.shape
        
        with tf.variable_scope("fully_con1") as scope:

            net = tf.reshape(conv4,[-1,35152])
            
            #net = tf.layers.flatten(net)
            print "flat"
            #print net.shape
            net = tf.layers.dense(inputs=net, units=32,activation=tf.nn.relu,reuse=reuse)
            print "dense1"
            #print net.shape
            net = tf.layers.dropout(inputs=net,rate=keep_rate,training=True)
            print "dropout"

        with tf.variable_scope("fully_con2") as scope:
            net =tf.layers.dense(inputs=net,units=16,reuse=reuse)
            print "dense final"

        with tf.variable_scope("fully_con3") as scope:
            net =tf.layers.dense(inputs=net,units=2,activation=tf.nn.tanh,reuse=reuse)
            print "dense final"
    return net


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
        tmp= y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
        return tf.reduce_mean(tmp + tmp2) /2
	    
