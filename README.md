# MNIST_experiments
my experiments of machine learning with MNIST data

some details on each of the files:

## cnn_mnist.py
a convolutinal neural network built using tf.layers according to the tutorial found at: https://www.tensorflow.org/tutorials/layers
i have added visualization of the first layer of filters, but with current settings i have yet to see them converge to smooth filters.

## tf_nn_train.py
a convolutional neural network built using tf.nn according to the same network setup as cnn_mnist.py. includes better visualization in tensorboard. 
model is found in /tf_nn_model/ . please note that msi files are not microsoft installers, this is just a result of tensorflow naming conventions for tensorboard, as that is the last 3 letters of my laptop name.


