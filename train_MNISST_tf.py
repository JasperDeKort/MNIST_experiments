# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:42:53 2017

@author: jaspe
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


print(mnist.validation)