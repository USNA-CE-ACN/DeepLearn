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

input_scales = [0.125,0.25,0.5,1,2,4,8]

input_file = sys.argv[1]
generate_analysis_file(input_file)
generate_json_file(input_file)

(ops,tensors,input_tensor) = parse_analysis_file()

input_name = input_file[input_file.rindex("/")+1:input_file.rindex(".")]
input_json = input_name + ".json"
json_ops = parse_json_file(input_json)

first_input = tensors[input_tensor]

for input_scale in input_scales:
    keras.backend.clear_session()
    scaledw = int(first_input[1] * input_scale)
    scaledh = int(first_input[2] * input_scale)
    scaled_tuple = (scaledw,scaledh,3)
    model_input = tf.keras.layers.Input(shape=scaled_tuple)
    set_first_input(scaled_tuple)
    resized_input = tf.keras.layers.Resizing(first_input[1],first_input[2])(model_input)
    first_op = Operation()
    first_op.layer_type = "INPUT"
    first_op.output_layer = resized_input
    ops[0].op_input = [first_op]
    
    model = create_model(ops,tensors,json_ops,model_input,0,[])
    output_model(model,str(input_scale) + "_" + input_name + ".tflite")
