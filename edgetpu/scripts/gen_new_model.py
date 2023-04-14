import sys,os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
import copy

from utility.create_model import *
from utility.parse_tf_json import *
from utility.parse_tf_analysis import *

input_file = sys.argv[1]
generate_analysis_file(input_file)
generate_json_file(input_file)

(ops,tensors,input_tensor) = parse_analysis_file()

input_name = input_file[input_file.rindex("/")+1:input_file.rindex(".")]
input_json = input_name + ".json"
json_ops = parse_json_file(input_json)
        
first_input = tensors[input_tensor]
set_first_input(first_input)

keras.backend.clear_session()

model_input = tf.keras.layers.Input(shape=first_input[1:])
first_op = Operation()
first_op.layer_type = "INPUT"
first_op.output_layer = model_input
ops[0].op_input = [first_op]

for onum in range(0,len(ops)):
    print(str(onum) + ": " + ops[onum].layer_type)

firstLayerRep = int(input("\nEnter the number of the first layer you want to replicate: "))
lastLayerRep = int(input("Enter the number of the last layer you want to replicate (same number as first if repeating single layer: "))

firstLayerRemove = int(input("Enter the number of the first layer you want to remove: "))
lastLayerRemove = int(input("Enter the number of the last layer you want to remove: "))

# First, extract the layers that we'll want to repeat
repeat_ops = ops[firstLayerRep:lastLayerRep+1]

front_ops = ops[0:firstLayerRemove]
back_ops = ops[lastLayerRemove+1:]

# This only works well with single input operations
repeat_ops[0].op_input = [front_ops[-1]]

ops = front_ops + back_ops
back_ops[0].op_input = [front_ops[-1]]

model = create_model(ops,tensors,json_ops,model_input)
output_model(model,"0_" + input_name + ".tflite")

# Fix back_ops for repeating
back_ops[0].op_input = [repeat_ops[-1]]

for i in range(1,21):
  keras.backend.clear_session()
  model_input = tf.keras.layers.Input(shape=first_input[1:])
  first_op = Operation()
  first_op.layer_type = "INPUT"
  first_op.output_layer = model_input

  iter_repeat_ops = []
  for j in range(0,i):
    repeat_ops[0].op_input = None
    next_ops = copy.deepcopy(repeat_ops)
    if j > 0:
      next_ops[0].op_input = [iter_repeat_ops[-1]]
    iter_repeat_ops = iter_repeat_ops + next_ops

  iter_repeat_ops[0].op_input = [front_ops[-1]]
  back_ops[0].op_input = [iter_repeat_ops[-1]]
  ops = front_ops + iter_repeat_ops + back_ops
  ops[0].op_input = [first_op]
  
  model = create_model(ops,tensors,json_ops,model_input)
  output_model(model,str(i) + "_" + input_name + ".tflite")
