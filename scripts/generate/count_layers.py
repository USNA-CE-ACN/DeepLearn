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

min_filters = 999999999
max_filters = 0
total_filters = 0
total_convs = 0

for op in ops:
    if "CONV_2D" in op.layer_type:
        filters = int(tensors[op.filter][0])
        min_filters = min(filters,min_filters)
        max_filters = max(filters,max_filters)
        total_filters = total_filters + filters
        total_convs = total_convs + 1

print("Min: " + str(min_filters))
print("Avg: " + str(total_filters/total_convs))
print("Max: " + str(max_filters))

