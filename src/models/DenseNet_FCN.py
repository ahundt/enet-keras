''' This file is under the MIT License.
'''
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from keras_contrib.applications import densenet
from keras_contrib.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
import keras.backend as K
import tensorflow as tf

def Atrous_DenseNet(input_shape=None, weight_decay=0.,
                    batch_momentum=0.9, batch_shape=None, classes=21):
    # TODO(ahundt) pass the parameters but use defaults for now
    densenet.DenseNet(depth=None, nb_dense_block=3, growth_rate=32,
                             nb_filter=-1, nb_layers_per_block=[6, 12, 24, 16],
                             bottleneck=True, reduction=0.5, dropout_rate=0.2,
                             weight_decay=1E-4,
                             include_top=True, top='segmentation',
                             weights=None, input_tensor=None,
                             input_shape=input_shape,
                             classes=classes, transition_dilation_rate=2,
                             transition_kernel_size=(1, 1),
                             transition_pooling=None)


def DenseNet_FCN(input_shape=None, weight_decay=0.,
                 batch_momentum=0.9, batch_shape=None, classes=21,
                 optimizer='adam', loss='categorical_crossentropy'):
    pixel_count = input_shape[0]*input_shape[1]
    shape_in = (input_shape[0], input_shape[1], 3)
    input_tensor = Input(shape=shape_in)
    x = densenet.__create_fcn_dense_net(classes,
                                        input_tensor,
                                        False,  # include_top
                                        input_shape=input_shape,
                                        nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                                        growth_rate=16,
                                        dropout_rate=0.2)

    x = Reshape((pixel_count, classes), input_shape=shape_in)(x)
    x = Activation('softmax')(x)
    name = 'DenseNet_FCN'
    model = Model(input_tensor, x, name=name)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', 'mean_squared_error'])
    return model, name
