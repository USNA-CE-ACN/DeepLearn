import sys,os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
import numpy as np
import re
import copy

import tensorflow_datasets as tfds

from utility.create_model import *
from utility.parse_tf_json import *
from utility.parse_tf_analysis import *

small_model_scales = [0.1,0.2,0.3,0.4,0.5]
large_model_scales = [0.9,0.8,0.7,0.6,0.5]
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

for i in range(len(small_model_scales)):
    small_model_scale = small_model_scales[i]
    large_model_scale = large_model_scales[i]
    #base_model = tf.keras.applications.MobileNet(
        #weights="imagenet",
    #    input_shape=(32,32,3),
    #    include_top=False,
    #    pooling="avg",
    #)  # Do not include the ImageNet classifier at the top. 

    base_model = tf.keras.applications.EfficientNetV2S(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(32,32,3),
        include_top=False,
        pooling="avg",
    )  # Do not include the ImageNet classifier at the top.

    base_model.summary()
    
    base_model.trainable = True
    
    inputs = keras.Input(shape=(32,32,3))

    last_small_layer=inputs
    last_large_layer=inputs

    new_small_layers = [inputs]
    new_large_layers = [inputs]

    output_to_layer = {}
    
    for layer in base_model.layers[1:]:
        output_to_layer[layer.output] = layer.name
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
            if len(last_small_layer.shape) > 3:
                small_layer = Reshape((1,last_small_layer.shape[3]))(last_small_layer)
                large_layer = Reshape((1,last_large_layer.shape[3]))(last_large_layer)
            elif len(last_small_layer.shape) > 2:
                small_layer = Reshape((1,last_small_layer.shape[2]))(last_small_layer)
                large_layer = Reshape((1,last_large_layer.shape[2]))(last_large_layer)
            else:
                small_layer = Reshape((1,1,last_small_layer.shape[1]))(last_small_layer)
                large_layer = Reshape((1,1,last_large_layer.shape[1]))(last_large_layer)
                
            large_layer.name = large_layer.name + "_L"
            
            last_small_layer = small_layer
            last_large_layer = large_layer
        elif isinstance(layer,Add):
            input_layers = [output_to_layer[lay] for lay in layer.input]
            positions = []
            i = 0
            for lay in base_model.layers:
                if lay.name in input_layers:
                    positions.append(i)
                i = i + 1

            last_small_layer = Add()([new_small_layers[p] for p in positions])
            last_large_layer = Add()([new_large_layers[p] for p in positions])
        elif isinstance(layer,Multiply):
            input_layers = [output_to_layer[lay] for lay in layer.input]
            positions = []
            i = 0
            for lay in base_model.layers:
                if lay.name in input_layers:
                    positions.append(i)
                i = i + 1
                        
            last_small_layer = Multiply()([new_small_layers[p] for p in positions])
            last_large_layer = Multiply()([new_large_layers[p] for p in positions])
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

        new_small_layers.append(last_small_layer)
        new_large_layers.append(last_large_layer)
            
    small_output = keras.layers.Dense(int(total_classes),activation="softmax")(last_small_layer)
    large_output = keras.layers.Dense(int(total_classes),activation="softmax")(last_large_layer)

    # TODO: FINAL OUTPUT LAYER HERE Add/Average
    output = keras.layers.Add()([small_output,large_output])
    #output = keras.layers.Average()([small_output,large_output])
    
    print("Creating Model Objects")
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
    
    print("Compiling and training model")
    
    full_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])                                                                                                                       
    full_model.fit(train_x,train_y, batch_size=256, epochs=100, validation_data=(test_x,test_y), shuffle=True)
    
    print("Evaluating model")
    
    scores = full_model.evaluate(test_x, test_y, verbose=1)                                                                                                                             
    print('Test loss:', scores[0])                                                                                                                                                                        
    print('Test accuracy:', scores[1])

