import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from os import getcwd


import make_parm_definitivo as make_parm
from model_elec2_1 import *

flags.DEFINE_integer('batch_size', 10, 'Batch size.') 
flags.DEFINE_integer('train_iter', 100, 'Total training iter')
flags.DEFINE_integer('step', 10, 'Save after ... iteration')

gen = make_parm.grid_maker(make_grid=False)

left = tf.placeholder(tf.float32, [None, 25,25,25, 1], name='left') 
right = tf.placeholder(tf.float32, [None, 25,25,25, 1], name='right')
with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label') 
    label = tf.to_float(label)

left_output = mynet(left, reuse=False)

right_output = mynet(right, reuse=True) 
                                        
margin = tf.Variable(1.2    ,trainable=True,name='margin')
loss = contrastive_loss(left_output, right_output, label, margin) 

global_step = tf.Variable(0, trainable=False) 

tf.summary.scalar('margin',margin)
starter_learning_rate = 3e-6
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96, staircase=True)
tf.summary.scalar('lr', learning_rate)

train_step = tf.train.RMSPropOptimizer(learning_rate,centered=True,name='RMSProp').minimize(loss, global_step=global_step)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print "treinando. biiirrlll!"
    
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
        
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('train.log', sess.graph)

    
    for i in range(FLAGS.train_iter):
        b_l, b_r, b_sim = gen.next_batch(10)

        _, l, summary_str = sess.run([train_step, loss, merged], 
            feed_dict={left:b_l, right:b_r, label: b_sim})
        
        writer.add_summary(summary_str, i)
        print "\r#%d - Loss"%i, l
        
        
            
            
            
            
            
            
            
            
            
            
            

    saver.save(sess, "model/model.ckpt")





