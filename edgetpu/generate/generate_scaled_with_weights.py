import sys,os
from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Reshape
import numpy as np
import re
import copy

import tensorflow_datasets as tfds

from utility.create_model import *
from utility.parse_tf_json import *
from utility.parse_tf_analysis import *

small_model_scale = 0.2
large_model_scale = 0.8

def copy_weights(old_layer,small_layer,large_layer):
    if(len(old_layer.get_weights()) > 2):
        print("WARNING: Extra weights!!!")
        arr = np.array(old_layer.get_weights()[1])
        print(arr.shape)

    small_shape = small_layer.get_weights()[0].shape
    large_shape = large_layer.get_weights()[0].shape
    
    # For now, using first N layers as small model weights and remaining M as large model weights
    # Hardcoded 3 parameter assumes image format (RGB)
    small_truncated_weights = tf.slice(old_layer.get_weights()[0],[0,0,0,0],[small_shape[0],small_shape[1],small_shape[2],small_shape[3]])
    large_truncated_weights = tf.slice(old_layer.get_weights()[0],[0,0,0,small_shape[3]],[large_shape[0],large_shape[1],large_shape[2],large_shape[3]])
    
    full_trunc_small = []
    full_trunc_small.append(small_truncated_weights)

    full_trunc_large = []
    full_trunc_large.append(large_truncated_weights)

    if(len(old_layer.get_weights()) > 1):
        small_truncated_weights = tf.slice(old_layer.get_weights()[1],[0],[small_shape[3]])
        large_truncated_weights = tf.slice(old_layer.get_weights()[1],[small_shape[3]],[large_shape[3]])
        
        full_trunc_small.append(small_truncated_weights)
        full_trunc_large.append(large_truncated_weights)
        
    small_layer.set_weights(full_trunc_small)
    large_layer.set_weights(full_trunc_large)

def load_data():
    result = tfds.load('imagenet2012', batch_size = -1)
    (x_train, y_train) = result['train']['image'],result['train']['label']
    (x_test, y_test) = result['test']['image'],result['test']['label']

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=1000)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=1000)
    return ((x_train, y_train), (x_test, y_test))

base_model = tf.keras.applications.MobileNet(
)  # Do not include the ImageNet classifier at the top. 

base_model.trainable = False

output_shape = base_model.get_layer(index=len(base_model.layers)-1).output_shape
inputs = keras.Input(shape=(224,224,3))

last_small_layer = inputs
last_large_layer = inputs

for layer in base_model.layers:
    if isinstance(layer,Conv2D):
        small_size_scaled = math.floor(layer.filters*small_model_scale)
        large_size_scaled = layer.filters - small_size_scaled
        small_layer_obj = Conv2D(small_size_scaled,layer.kernel_size,layer.strides,layer.padding,layer.data_format,layer.dilation_rate,layer.groups,layer.activation,layer.use_bias,layer.kernel_initializer,layer.bias_initializer,layer.kernel_regularizer,layer.bias_regularizer,layer.kernel_constraint,layer.bias_constraint)
        small_layer = small_layer_obj(last_small_layer)

        large_layer_obj = Conv2D(large_size_scaled,layer.kernel_size,layer.strides,layer.padding,layer.data_format,layer.dilation_rate,layer.groups,layer.activation,layer.use_bias,layer.kernel_initializer,layer.bias_initializer,layer.kernel_regularizer,layer.bias_regularizer,layer.kernel_constraint,layer.bias_constraint)
        large_layer = large_layer_obj(last_large_layer)

        copy_weights(layer,small_layer_obj,large_layer_obj)
        
        last_small_layer = small_layer
        last_large_layer = large_layer
    elif isinstance(layer,DepthwiseConv2D) and layer.filters != None:
        small_size_scaled = math.floor(layer.filters*small_model_scale)
        large_size_scaled = layer.filters - small_size_scaled

        small_layer_obj = DepthwiseConv2D(small_size_scaled,layer.kernel_size,layer.strides,layer.padding,layer.depth_multiplier,layer.data_format,layer.dilation_rate,layer.activation,layer.use_bias,layer.depthwise_initializer,layer.bias_initializer,layer.depthwise_regularizer,layer.bias_regularizer,layer.activity_regularizer,layer.depthwise_constraint,layer.bias_constraint)
        small_layer = small_layer_obj(last_small_layer)

        large_layer_obj = DepthwiseConv2D(large_size_scaled,layer.kernel_size,layer.strides,layer.padding,layer.depth_multiplier,layer.data_format,layer.dilation_rate,layer.activation,layer.use_bias,layer.depthwise_initializer,layer.bias_initializer,layer.depthwise_regularizer,layer.bias_regularizer,layer.activity_regularizer,layer.depthwise_constraint,layer.bias_constraint)
        large_layer = large_layer_obj(last_large_layer)

        copy_weights(layer,small_layer_obj,large_layer_obj)
        
        last_small_layer = small_layer
        last_large_layer = large_layer
    elif isinstance(layer,Reshape):
        small_layer = Reshape((1,last_small_layer.shape[3]))(last_small_layer)
        large_layer = Reshape((1,last_large_layer.shape[3]))(last_large_layer)

        last_small_layer = small_layer
        last_large_layer = large_layer
    else:
        config = layer.get_config()
        weights = layer.get_weights()
        cloned_small_layer = type(layer).from_config(config)(last_small_layer)
        cloned_large_layer = type(layer).from_config(config)(last_large_layer)
        
        last_small_layer = cloned_small_layer
        last_large_layer = cloned_large_layer

small_output = keras.layers.Dense(last_small_layer.shape[2],activation="softmax")(last_small_layer)
large_output = keras.layers.Dense(last_large_layer.shape[2],activation="softmax")(last_large_layer)

small_model = keras.Model(inputs,small_output)
large_model = keras.Model(inputs,large_output)

(x_train, y_train), (x_test, y_test) = load_data()

sgd = tf.keras.optimizers.SGD(
    learning_rate = 0.1,
    momentum = 0.9,
    nesterov=True,
    weight_decay=1e-5)

small_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])                                                                                                                       
#small_model.fit(x_train, y_train, batch_size=256, epochs=25, validation_data=(x_test,y_test), shuffle=True)                                                                                                     
scores = small_model.evaluate(x_test, y_test, verbose=1)                                                                                                                                                        
print('Test loss:', scores[0])                                                                                                                                                                            
print('Test accuracy:', scores[1])

large_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])                                                                                                                       
#large_model.fit(x_train, y_train, batch_size=256, epochs=25, validation_data=(x_test,y_test), shuffle=True)                                                                                                     
scores = large_model.evaluate(x_test, y_test, verbose=1)                                                                                                                                                        
print('Test loss:', scores[0])                                                                                                                                                                            
print('Test accuracy:', scores[1])

