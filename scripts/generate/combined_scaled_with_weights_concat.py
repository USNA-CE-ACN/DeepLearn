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

small_model_scale = 0.2
large_model_scale = 0.8
total_classes = 10

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
    first_n_labels = []
    result = tfds.load('cifar10', batch_size=-1)
    print("Finished load")
    (x_train, y_train) = result['train']['image'],result['train']['label']
    (x_test, y_test) = result['test']['image'],result['test']['label']

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    for i in range(len(x_train)):
        image = x_train[i]
        train_x.append(image)
        train_y.append(y_train[i])

    for i in range(len(x_test)):
        image = x_test[i]
        test_x.append(image)
        test_y.append(y_test[i])

    x_train = tf.convert_to_tensor(train_x,dtype=tf.uint8)
    x_test = tf.convert_to_tensor(test_x,dtype=tf.uint8)

    train_y = tf.convert_to_tensor(train_y)
    test_y = tf.convert_to_tensor(test_y)
    
    y_train = tf.keras.utils.to_categorical(train_y, num_classes=int(total_classes))
    y_test = tf.keras.utils.to_categorical(test_y, num_classes=int(total_classes))
    
    return (x_train, y_train, x_test, y_test)

base_model = tf.keras.applications.MobileNet(
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

last_small_layer=inputs
last_large_layer=inputs

for layer in base_model.layers[1:]:
    if isinstance(layer,Conv2D):
        small_size_scaled = math.floor(layer.filters*small_model_scale)
        large_size_scaled = layer.filters - small_size_scaled
        small_layer_obj = Conv2D(small_size_scaled,layer.kernel_size,layer.strides,layer.padding,layer.data_format,layer.dilation_rate,layer.groups,layer.activation,layer.use_bias,layer.kernel_initializer,layer.bias_initializer,layer.kernel_regularizer,layer.bias_regularizer,layer.kernel_constraint,layer.bias_constraint)
        small_layer = small_layer_obj(last_small_layer)

        large_layer_obj = Conv2D(large_size_scaled,layer.kernel_size,layer.strides,layer.padding,layer.data_format,layer.dilation_rate,layer.groups,layer.activation,layer.use_bias,layer.kernel_initializer,layer.bias_initializer,layer.kernel_regularizer,layer.bias_regularizer,layer.kernel_constraint,layer.bias_constraint)
        large_layer = large_layer_obj(last_large_layer)
        large_layer_obj.name = large_layer_obj.name + "_L"
        large_layer.name = large_layer.name + "_L"

        copy_weights(layer,small_layer_obj,large_layer_obj)
        
        last_small_layer = small_layer
        last_large_layer = large_layer
    elif isinstance(layer,Reshape):
        small_layer = Reshape((1,last_small_layer.shape[3]))(last_small_layer)
        large_layer = Reshape((1,last_large_layer.shape[3]))(last_large_layer)

        large_layer.name = large_layer.name + "_L"

        last_small_layer = small_layer
        last_large_layer = large_layer
    else:
        config = layer.get_config()
        weights = layer.get_weights()
        cloned_small_layer = layer.__class__.from_config(config)(last_small_layer)
        cloned_large_obj = layer.__class__.from_config(config)
        cloned_large_layer = cloned_large_obj(last_large_layer)

        cloned_large_obj.name = cloned_large_obj.name + "_L"
        cloned_large_layer.name = cloned_large_layer.name + "_L"
        
        last_small_layer = cloned_small_layer
        last_large_layer = cloned_large_layer


concat = keras.layers.Concatenate()([last_small_layer,last_large_layer])
output = keras.layers.Dense(total_classes,activation="softmax")(concat)
        
#small_output = keras.layers.Dense(int(total_classes*small_model_scale),activation="softmax")(last_small_layer)
#large_output = keras.layers.Dense(int(total_classes*large_model_scale),activation="softmax")(last_large_layer)

#concat = keras.layers.Concatenate()([small_output,large_output])

print("Creating Model Objects")
#full_model = keras.Model(inputs,concat)
full_model = keras.Model(inputs,output)

full_model.summary()

print("Loading Data")

(train_x,train_y,test_x, test_y) = load_data()

print("Creating Opt")

sgd = tf.keras.optimizers.SGD(
    learning_rate = 0.1,
    momentum = 0.9,
    nesterov=True,
    weight_decay=1e-5)

print("Compiling model")

full_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])                                                                                                                       
full_model.fit(train_x,train_y, batch_size=256, epochs=100, validation_data=(test_x,test_y), shuffle=True)

print("Evaluating large model")

scores = full_model.evaluate(test_x, test_y, verbose=1)                                                                                                                             
print('Test loss:', scores[0])                                                                                                                                                                            
print('Test accuracy:', scores[1])

