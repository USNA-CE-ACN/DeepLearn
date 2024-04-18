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

input_scales = [.1,.2,.3,.7,.8,.9]

input_file = sys.argv[1]
generate_analysis_file(input_file)
generate_json_file(input_file)

(ops,tensors,input_tensor) = parse_analysis_file()

input_name = input_file[input_file.rindex("/")+1:input_file.rindex(".")]
input_json = input_name + ".json"
json_ops = parse_json_file(input_json)

first_input = tensors[input_tensor]

first_conv_op = None

for op in ops:
    if "CONV_2D" in op.layer_type:
        first_conv_op = op
        break

print(tensors[first_conv_op.output])
print(np.prod(tensors[first_conv_op.output]))

for input_scale in input_scales:
    keras.backend.clear_session()

    # Get the total size of the output parameters from the first convolution operation
    output_total = np.prod(tensors[first_conv_op.output])
    output_scaled = input_scale * output_total

    input_total = np.prod(tensors[first_conv_op.input_num[0]])
    op_factor = output_total / input_total

    output_factored = (output_scaled / op_factor) / 3 # TODO: Fix, assumes NxNx3 tensor

    scaled_M = math.sqrt(output_factored)
    real_input_scale = scaled_M / first_input[1]
    
    scaledw = int(first_input[1] * real_input_scale)
    scaledh = int(first_input[2] * real_input_scale)
    scaled_tuple = (scaledw,scaledh,3)
    model_input = tf.keras.layers.Input(shape=scaled_tuple,batch_size=1)
    set_first_input((1,scaledw,scaledh,3))
    #resized_input = tf.keras.layers.Resizing(first_input[1],first_input[2])(model_input)
    first_op = Operation()
    first_op.layer_type = "INPUT"
    first_op.output_layer = model_input
    ops[0].op_input = [first_op]
    
    model = create_model(ops,tensors,json_ops,model_input,0)
    output_model(model,str(input_scale) + "_" + input_name + ".tflite")
