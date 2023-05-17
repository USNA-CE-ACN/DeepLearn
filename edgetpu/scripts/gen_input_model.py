import sys,os
from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
import copy

from utility.create_model import *
from utility.parse_tf_json import *
from utility.parse_tf_analysis import *

input_sizes = [10,100,1000,10000]

input_file = sys.argv[1]
generate_analysis_file(input_file)
generate_json_file(input_file)

(ops,tensors,input_tensor) = parse_analysis_file()

input_name = input_file[input_file.rindex("/")+1:input_file.rindex(".")]
input_json = input_name + ".json"
json_ops = parse_json_file(input_json)
        
first_input = tensors[input_tensor]

for input_size in input_sizes:
    keras.backend.clear_session()
    model_input = tf.keras.layers.Input(shape=(input_size))
    set_first_input((input_size,))
    dense_input = tf.keras.layers.Dense(math.prod(first_input[1:]))(model_input)
    shaped_input = tf.keras.layers.Reshape(first_input[1:])(dense_input)
    #resized_input = tf.keras.layers.Resizing(first_input[1],first_input[2])(shaped_input)
    #shaped_input = tf.keras.layers.Reshape(first_input[1:])(dense_input)
    first_op = Operation()
    first_op.layer_type = "INPUT"
    first_op.output_layer = shaped_input
    ops[0].op_input = [first_op]
    
    model = create_model(ops,tensors,json_ops,model_input,0,[])
    output_model(model,str(input_size) + "_" + input_name + ".tflite")
