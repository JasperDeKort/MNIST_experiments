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

log_folder = "./tf_nn_model2"

inputsize = [28, 28, 1]

# tweakable parameters
l2beta = 0.03 # only if l2 loss is added
epsilon = 1
learning_rate = 0.03

input_keep = 0.8
layer_keep = 0.4
filtersize= 5
l1_outputchan = 32
l2_outputchan = 64
finallayer_in = 3136
denselayernodes = 1024

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
weights1 = init_weights([finallayer_in,denselayernodes],"weights_1")
weights2 = init_weights([denselayernodes,n_classes],"weights_2")

tf.summary.histogram("weights_1", weights1)
tf.summary.histogram("weights_2", weights2)

# tensors for calculation of summaries and visualizations
identity10 = tf.eye(10, dtype='float32')

# dimensions of input and output
x = tf.placeholder('float', [None , 784], name='input_data')
y = tf.placeholder('float', [None, 10], name='output_data')
layer_keep_holder = tf.placeholder("float", name="input_keep")

prediction = model(x, filter1,filter2 , layer_keep_holder, weights1, weights2)

with tf.name_scope("cost"):
    #regularization = tf.nn.l2_loss(filter1)
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
    #cost = loss + l2beta * regularization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=epsilon).minimize(cost)

#generate summaries in test name scope to collect and merge easily        
with tf.name_scope("test"):
    with tf.name_scope("cost"):
        tf.summary.scalar("cost", cost)
    
    # generate summary of multiple accuracy metrics    
    with tf.name_scope("accuracy"):
        test_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prediction,1)
        correct_pred = tf.equal(test_y, pred_y) # Count correct predictions
        acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
        tf.summary.scalar("accuracy", acc_op)
        conf_mat = tf.confusion_matrix(test_y, pred_y, dtype='int32')
        conf_mat2 = tf.confusion_matrix(test_y, pred_y, dtype='float32')
        zerodiag = conf_mat2 - tf.multiply(conf_mat2, identity10)
        conf_mat_reshaped = tf.reshape(zerodiag, [1,10,10,1])
        tf.summary.image('convolution_matrix',conf_mat_reshaped)

# generate summaries in train name scope to collect and merge easily        
with tf.name_scope("train"):
    with tf.name_scope("cost"):
        tf.summary.scalar("cost", cost)
    
    # generate summary of multiple accuracy metrics    
    with tf.name_scope("accuracy"):
        tf.summary.scalar("accuracy", acc_op)
        
# generate images of the filters for human viewing
with tf.variable_scope('visualization_filter1'):
    # to tf.image_summary format [batch_size, height, width, channels]
    kernel_transposed = tf.transpose (filter1, [3, 0, 1, 2])
    # reshape from 2 channel filters to 1 channel filters for image gen
    tf.summary.image('conv1/filters', kernel_transposed, max_outputs=32)

# generate images from filter pass through results. 1 for each class.        
with tf.name_scope("f1pass"):
    imageconvlist = []
    data = tf.reshape(x,[-1,28,28,1])
    imageconv1 = tf.nn.relu(tf.nn.conv2d(data, filter1,strides=[1,1,1,1],padding="SAME"))
    for i in range(n_classes):
        imageconvlist.append(tf.transpose(tf.reshape(imageconv1[i],[1,28,28,32]), [3,1,2,0]))        
        tf.summary.image('number_{}'.format(i), imageconvlist[i],max_outputs = 32)

with tf.name_scope("f2pass"):
    imageconvlist2 = []
    imagepool1 = tf.layers.max_pooling2d(inputs=imageconv1, pool_size=[2 , 2], strides=2)
    imageconv2 = tf.nn.relu(tf.nn.conv2d(imagepool1, filter2,strides=[1,1,1,1],padding="SAME"))
    for i in range(n_classes):
        imageconvlist2.append(tf.transpose(tf.reshape(imageconv2[i],[1,14,14,64]), [3,1,2,0]))
        tf.summary.image('number_{}'.format(i), imageconvlist2[i],max_outputs = 64)
       
def train_neural_network(x_train, y_train, x_test, y_test,x_img,y_img):
    print('test set size: {}'.format(len(y_test)))
    print('train set size: {}'.format(len(y_train)))
    # number of cycles of feed forward and back propagation
    hm_epochs = 25
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, )
    i = 0
    print('starting training')
    with tf.Session() as sess:   
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_folder, sess.graph)
        # generate 2 summary merges. one for use with test data, one for use with train data.
        # data uninfluenced by test or train data is added to the train summary.
        trainmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train') + \
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='visualization_filter1'))
        testmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='test'))
        imagepassmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='f1pass') + \
                                          tf.get_collection(tf.GraphKeys.SUMMARIES, scope='f2pass'))
        
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
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y,
                                                                layer_keep_holder: layer_keep})
                epoch_loss += c
                i += end - start
                #generate summaries every 10000 examples
                if start % 10000 == 0:
                    testsummary= sess.run(testmerge, feed_dict={x: x_test, y: y_test, layer_keep_holder: 1})
                    writer.add_summary(testsummary,i)                   
                    trainsummary = sess.run(trainmerge, feed_dict={x: x_train[:10000], y: y_train[:10000], layer_keep_holder: 1} )
                    writer.add_summary(trainsummary,i)
            summary, acc, conf = sess.run([testmerge, acc_op, conf_mat], feed_dict={x: x_test, y: y_test, layer_keep_holder: 1})
            print('Epoch ', epoch + 1, ' completed out of ', hm_epochs, ' ,epoch loss: ', epoch_loss)
            print('current accuracy on test  set {}'.format(acc))
            acctrain = sess.run(acc_op, feed_dict={x: x_train[:10000], y: y_train[:10000], layer_keep_holder: 1})
            print('current accuracy on train set {}'.format(acctrain))
            print(conf)
            # generate images by filter pass through for filter 1 and 2
            imagesum = sess.run(imagepassmerge,feed_dict={x: x_img})
            writer.add_summary(imagesum,i)
            saver.save(sess, log_folder + '/v01', global_step=i)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Final accuracy: ', accuracy.eval({x: x_test, 
                                                 y: y_test,
                                                 layer_keep_holder: 1}) )
        saver.save(sess, log_folder + '/v01', global_step=i)
        writer.flush()
        writer.close()
        
def pick_image_for_class(x_test,y_test, classes):
    x_images = []
    y_images = []   
    for i in range(classes):
        index = np.where(np.argmax(y_test,1)==i)[0][0]
        x_images.append( x_test[index])
        y_images.append( i)
    x_images = np.array(x_images)
    return x_images, y_images

def main():
    print('loading data')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x_train = mnist.train.images  # Returns np.array
    y_train = np.asarray(mnist.train.labels, dtype=np.int32)
    x_test = mnist.test.images  # Returns np.array
    y_test = np.asarray(mnist.test.labels, dtype=np.int32)
    x_img, y_img = pick_image_for_class(x_test,y_test,10)
    print('data loaded, starting training')
    train_neural_network(x_train, y_train, x_test, y_test,x_img,y_img)
     

if __name__ == "__main__":
    main()