# MNIST_experiments
my experiments of machine learning with MNIST data

some details on each of the files:

### cnn_mnist.py
a convolutinal neural network built using tf.layers according to the tutorial found at: https://www.tensorflow.org/tutorials/layers
i have added visualization of the first layer of filters, but with current settings i have yet to see them converge to smooth filters.

### tf_nn_train.py
a convolutional neural network built using tf.nn according to the same network setup as cnn_mnist.py. includes better visualization in tensorboard. filter 1 is visualized directly and by passing through an image for each class. filter 2 is visualized only by passing through an image for each class.
warning: this generates a large amount of images in tensorboard. i reccomend using regular expression filtering to only show the desired images


