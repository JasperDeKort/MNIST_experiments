# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:42:53 2017

@author: jaspe
"""

import tensorflow as tf
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data


tf.reset_default_graph()

n_classes = 10
batch_size = 100

log_folder = "./tf.nn_model"

inputsize = [28, 28, 1]

# tweakable parameters
l2beta = 0.03
epsilon = 1
learning_rate = 0.03

input_keep = 0.8
layer_keep = 0.4
filtersize= 5
l1_outputchan = 32
l2_outputchan = 64
finallayer_in = 3136

def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.04, name=name), name=name)

def model(input_data, filter1,filter2 , layer_keep, weights1, weights2):
    
    data = tf.reshape(input_data, [-1,28,28,1])
    # Add layer name scopes for better graph visualization
    with tf.name_scope("hidden_1_conv"):
        conv1 = tf.nn.relu(tf.nn.conv2d(data, filter1,strides=[1,1,1,1],padding="SAME"))
    with tf.name_scope("max_pooling_layer_1"):
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2 , 2], strides=2)
    with tf.name_scope("hidden_2_conv"):
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, filter2,strides=[1,1,1,1],padding="SAME"))
    with tf.name_scope("max_pooling_layer_2"):
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    with tf.name_scope("hidden_3_dense"):
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        layer3 = tf.nn.relu(tf.matmul(pool2_flat, weights1))
        dropout = tf.nn.dropout(layer3, layer_keep)
    with tf.name_scope("hidden_4_dense"):    
        output = tf.nn.relu(tf.matmul(dropout, weights2))
        return output

# define filters and weights
filter1 = init_weights([filtersize,filtersize,inputsize[2], l1_outputchan], "filter_1")
filter2 = init_weights([filtersize,filtersize,l1_outputchan, l2_outputchan], "filter_2")
weights1 = init_weights([finallayer_in,1024],"weights_1")
weights2 = init_weights([1024,n_classes],"weights_2")

tf.summary.histogram("weights_1", weights1)
tf.summary.histogram("weights_2", weights2)

# dimensions of input and output
x = tf.placeholder('float', [None , 784], name='input_data')
y = tf.placeholder('float', [None, 10], name='output_data')
layer_keep_holder = tf.placeholder("float", name="input_keep")

prediction = model(x, filter1,filter2 , layer_keep_holder, weights1, weights2)

with tf.name_scope("cost"):
    # optimize, learning_rate = 0.001
    #regularization = tf.nn.l2_loss(filter1)
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
    #cost = loss + l2beta * regularization
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=epsilon).minimize(cost)
    tf.summary.scalar("cost", cost)

# generate summary of multiple accuracy metrics    
with tf.name_scope("accuracy"):
    test_y = tf.argmax(y, 1)
    pred_y = tf.argmax(prediction,1)
    correct_pred = tf.equal(test_y, pred_y) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    tf.summary.scalar("accuracy", acc_op)
    conf_mat = tf.confusion_matrix(test_y, pred_y, dtype="uint8")
    conf_mat_reshaped = tf.reshape(conf_mat, [1,10,10,1])
    tf.summary.image('convolution matrix',conf_mat_reshaped)

# generate images of the filters for human viewing
with tf.variable_scope('visualization_filter1'):
    # to tf.image_summary format [batch_size, height, width, channels]
    kernel_transposed = tf.transpose (filter1, [3, 0, 1, 2])
    # reshape from 2 channel filters to 1 channel filters for image gen
    tf.summary.image('conv1/filters', kernel_transposed, max_outputs=32)

# generate over time summary of min, max and mean predicted values for each class
with tf.variable_scope('output_values'):
    class_max = tf.reduce_max(prediction, reduction_indices=[0])
    class_min = tf.reduce_min(prediction, reduction_indices=[0])
    class_mean = tf.reduce_mean(prediction, reduction_indices=[0])
    tf.summary.scalar('class_0_max', class_max[0])
    tf.summary.scalar('class_1_max', class_max[1])
    tf.summary.scalar('class_0_min', class_min[0])
    tf.summary.scalar('class_1_min', class_min[1])
    tf.summary.scalar('class_0_mean', class_mean[0])
    tf.summary.scalar('class_1_mean', class_mean[1])
    
    
def train_neural_network(x_train, y_train, x_test, y_test):
    print('test set size: {}'.format(len(y_test)))
    print('train set size: {}'.format(len(y_train)))
    # number of cycles of feed forward and back propagation
    hm_epochs = 100
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, )
    i = 0
    print('starting training')
    with tf.Session() as sess:   
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_folder, sess.graph) # for 1.0
        merged = tf.summary.merge_all()
        
        if os.path.isfile( log_folder + '/checkpoint'):
            print('previous version found. continguing')
            saver.restore(sess,tf.train.latest_checkpoint(log_folder))
            #read checkpoint file and cast number at the end to int
            ckpt = tf.train.get_checkpoint_state(log_folder)
            i = int(str(ckpt).split('-')[-1][:-2])

            
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train)+1, batch_size)):
                if end > len(x_train):
                    end = len(x_train)
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])
                #print('training samples {} to {}'.format(start, end))
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y,
                                                                layer_keep_holder: layer_keep})
                epoch_loss += c
                i += end - start
                if start % 10000 == 0:
                    summary, acc = sess.run([merged, acc_op], feed_dict={x: x_test, y: y_test,
                                                                    layer_keep_holder: 1})
                    writer.add_summary(summary,i)
                    print('current accuracy: {} at step {}'.format(acc, end))
            summary, acc, conf = sess.run([merged, acc_op, conf_mat], feed_dict={x: x_test, y: y_test, layer_keep_holder: 1})
            print('Epoch ', epoch + 1, ' completed out of ', hm_epochs, ' ,epoch loss: ', epoch_loss)
            print(conf)
            saver.save(sess, log_folder + '/v01', global_step=i)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Final accuracy: ', accuracy.eval({x: x_test, 
                                                 y: y_test,
                                                 layer_keep_holder: 1}) )
        saver.save(sess, log_folder + '/v01', global_step=i)
        writer.flush()
        writer.close()


def main():
    print('loading data')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = mnist.train.images  # Returns np.array
    y_train = np.asarray(mnist.train.labels, dtype=np.int32)
    x_test = mnist.test.images  # Returns np.array
    y_test = np.asarray(mnist.test.labels, dtype=np.int32)
    print('data loaded, starting training')
    train_neural_network(x_train, y_train, x_test, y_test)
     

if __name__ == "__main__":
    main()