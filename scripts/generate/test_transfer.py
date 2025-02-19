import sys,os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
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

total_classes = 10
large_model_scale = 1

def copy_weights(old_layer,large_layer):
    if(len(old_layer.get_weights()) > 2):
        print("WARNING: Extra weights!!!")
        arr = np.array(old_layer.get_weights()[1])
        print(arr.shape)

    large_shape = large_layer.get_weights()[0].shape
    
    # For now, using first N layers as small model weights and remaining M as large model weights
    # Hardcoded 3 parameter assumes image format (RGB)
    large_truncated_weights = tf.slice(old_layer.get_weights()[0],[0,0,0,0],[large_shape[0],large_shape[1],large_shape[2],large_shape[3]])

    full_trunc_large = []
    full_trunc_large.append(large_truncated_weights)

    if(len(old_layer.get_weights()) > 1):
        large_truncated_weights = tf.slice(old_layer.get_weights()[1],[0],[large_shape[3]])
        full_trunc_large.append(large_truncated_weights)
        
    large_layer.set_weights(full_trunc_large)
 
def load_data():
    result = tfds.load('cifar10', batch_size=-1)
    print("Finished load")
    (x_train, y_train) = result['train']['image'],result['train']['label']
    (x_test, y_test) = result['test']['image'],result['test']['label']

    large_train_x = []
    large_train_y = []
    large_test_x = []
    large_test_y = []

    for i in range(len(x_train)):
        image = x_train[i]
        if y_train[i] < int(total_classes * large_model_scale):
            large_train_x.append(image)
            large_train_y.append(y_train[i])
    
    for i in range(len(x_test)):
        image = x_test[i]
        if y_test[i] < int(total_classes * large_model_scale):
            large_test_x.append(image)
            large_test_y.append(y_test[i])

    large_train_x = tf.convert_to_tensor(large_train_x,dtype=tf.uint8)
    large_test_x = tf.convert_to_tensor(large_test_x,dtype=tf.uint8)
    large_train_y = tf.convert_to_tensor(large_train_y)
    large_test_y = tf.convert_to_tensor(large_test_y)

    large_train_y = tf.keras.utils.to_categorical(large_train_y, num_classes=int(total_classes * large_model_scale))
    large_test_y = tf.keras.utils.to_categorical(large_test_y, num_classes=int(total_classes * large_model_scale))
    
    return (large_train_x, large_train_y, large_test_x,large_test_y)

base_model = tf.keras.applications.EfficientNetV2S(
    #weights="imagenet",
    input_shape=(32,32,3),
    include_top=False,
    pooling="avg",
)  # Do not include the ImageNet classifier at the top. 

base_model.trainable = True

inputs = keras.Input(shape=(32,32,3))

#connect_input_layer = keras.layers.Resizing(224,224)(inputs)

#last_small_layer = connect_input_layer
#last_large_layer = connect_input_layer

last_large_layer=inputs

for layer in base_model.layers[1:]:
    if isinstance(layer,Conv2D):
        large_size_scaled = layer.filters
        large_layer_obj = Conv2D(large_size_scaled,layer.kernel_size,layer.strides,layer.padding,layer.data_format,layer.dilation_rate,layer.groups,layer.activation,layer.use_bias,layer.kernel_initializer,layer.bias_initializer,layer.kernel_regularizer,layer.bias_regularizer,layer.kernel_constraint,layer.bias_constraint)
        large_layer = large_layer_obj(last_large_layer)

        copy_weights(layer,large_layer_obj)
        
        last_large_layer = large_layer
    elif isinstance(layer,Reshape):
        large_layer = Reshape((1,last_large_layer.shape[3]))(last_large_layer)

        last_large_layer = large_layer
    else:
        config = layer.get_config()
        weights = layer.get_weights()
        cloned_large_layer = layer.__class__.from_config(config)(last_large_layer)
        
        last_large_layer = cloned_large_layer
        
large_output = keras.layers.Dense(int(total_classes*large_model_scale),activation="softmax")(last_large_layer)

print("Creating Model Objects")

large_model = keras.Model(inputs,large_output)

print("Loading Data")

(large_train_x,large_train_y,large_test_x, large_test_y) = load_data()

print("Creating Opt")

sgd = tf.keras.optimizers.SGD(
    learning_rate = 0.1,
    momentum = 0.9,
    nesterov=True,
    weight_decay=1e-5)

print("Compiling large model")

large_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])                                                                                                                       
large_model.fit(large_train_x,large_train_y, batch_size=256, epochs=100, validation_data=(large_test_x,large_test_y), shuffle=True)

print("Evaluating large model")

scores = large_model.evaluate(large_test_x, large_test_y, verbose=1)                                                                                                                                            
print('Test loss:', scores[0])                                                                                                                                                                            
print('Test accuracy:', scores[1])

