# This script creates a convolutional neural network model with a
# specified input size and number of convolutional layers, all with
# the same filter and bias size

# Input in the form python3 gen_simple [input_size] [output_size] [num_layers] [bias_size] [filter_size]
# e.g. python3 gen_simple.py 224 1001 5 64 3

import sys,os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
import copy

# You may want to change how inputs are listed...
input_size = int(sys.argv[1])
output_size = int(sys.argv[2])
num_layers = int(sys.argv[3])
bias_size = int(sys.argv[4])
filter_size = int(sys.argv[5])

input_shape = (1,input_size,input_size,3)
model_input = tf.keras.layers.Input(shape=input_shape[1:],batch_size=1)
prev_op = model_input

first_input = input_shape

for i in range(0,num_layers):
    layer = tf.keras.layers.Conv2D(bias_size,(filter_size,filter_size),
                                   (1,1),
                                   activation="relu",
                                   padding="same",
                                   use_bias=False,
                                   input_shape=prev_op.shape)(prev_op)
    prev_op = layer # Chain to next layer

print(layer.shape)
    
pool2d = tf.keras.layers.AveragePooling2D(pool_size=(layer.shape[1],layer.shape[2]),
                                          strides=(2,2),padding="valid",
                                          input_shape=layer.shape)(layer)

conv2d = tf.keras.layers.Conv2D(output_size, (1,1), (1,1), padding="same",
                                input_shape=pool2d.shape)(pool2d)

reshape = tf.keras.layers.Reshape((1,output_size), input_shape=conv2d.shape)(conv2d) 

softmax = tf.keras.layers.Softmax()(reshape)

model = tf.keras.models.Model(inputs=model_input,outputs=softmax)

adam = keras.optimizers.Adam(epsilon = 1e-08)
#model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.compile()
print("Compiled model")

#model.save(filename + ".keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model) #this works!!!!

def representative_data_gen():
    X_full = np.random.rand(*((100,) + first_input))
    for i in range(3):
        yield [X_full[i].astype(np.float32)]

print("Created converter ")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
#    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

print("Converted model")

with open("model.tflite", 'wb') as f:
    f.write(tflite_model)
