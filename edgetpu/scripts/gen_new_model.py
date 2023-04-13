import sys,os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

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

input_size = (100,)
X_full = np.random.rand(*(input_size + first_input))
rng = np.random.default_rng()
Y_full = rng.integers(0,1000,100)

def representative_data_gen():
  for i in range(100):
    yield [X_full[i].astype(np.float32)]

keras.backend.clear_session()

a = tf.keras.layers.Input(shape=first_input[1:])
first_op = Operation()
first_op.layer_type = "INPUT"
first_op.output_layer = a
ops[0].op_input.append(first_op)

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
back_ops[0].op_input = [repeat_ops[-1]]

print("READY")
print_op(front_ops[-1])
print_op(repeat_ops[0])
print_op(repeat_ops[-1])
print_op(back_ops[0])
print("SET")
         
ops = front_ops + repeat_ops + back_ops

# Now, clone layers at the insertion point and generate models
for op in ops:
  print_op(op)
  if op.layer_type == "CONV_2D" or op.layer_type == "DEPTHWISE_CONV_2D":
    input = op.op_input[0].output_layer
    filter = tensors[op.filter]
    json_op = json_ops[op.output]
    bops = json_op["builtin_options"]
    act = None
    if bops["fused_activation_function"] != "NONE":
      act = bops["fused_activation_function"].lower()
    if op.layer_type == "CONV_2D":
      layer = tf.keras.layers.Conv2D(filter[0],(filter[1],filter[2]),
                                     (bops["stride_w"],bops["stride_h"]),
                                     padding=bops["padding"].lower(),
                                     activation=act,
                                     input_shape=input.shape)(input)
    else:
      layer = tf.keras.layers.DepthwiseConv2D(filter[0],(bops["stride_w"],bops["stride_h"]),
                                              padding=bops["padding"].lower(),
                                              depth_multiplier=bops["depth_multiplier"],
                                              activation=act,
                                              input_shape=input.shape)(input)
    op.output_layer = layer
  elif op.layer_type == "FULLY_CONNECTED":
    input = op.op_input[0].output_layer
    dim = filter[0]
    json_op = json_ops[op.output]
    bops = json_op["builtin_options"]
    act = None
    if bops["fused_activation_function"] != "NONE":
      act = bops["fused_activation_function"].lower()
    layer = tf.keras.layers.Dense(dim,act)(input)
    op.output_layer = layer
  elif op.layer_type == "CONCATENATION":
    concat_list = []
    for input_op in op.op_input:
      concat_list.append(input_op.output_layer)
    layer = tf.keras.layers.Concatenate()(concat_list)
    op.output_layer = layer
  elif op.layer_type == "ADD":
    add_list = []
    for input_op in op.op_input:
      add_list.append(input_op.output_layer)
    layer = tf.keras.layers.Add()(add_list)
    op.output_layer = layer
  elif op.layer_type == "MEAN":
    input = op.op_input[0].output_layer
    layer = tf.keras.layers.GlobalAveragePooling2D()(input)
    op.output_layer = layer
  elif op.layer_type == "MAX_POOL_2D":
    input = op.op_input[0].output_layer
    json_op = json_ops[op.output]
    bops = json_op["builtin_options"]
    layer = tf.keras.layers.MaxPool2D((bops["filter_width"],bops["filter_height"]),
                                      (bops["stride_w"],bops["stride_h"]),
                                      padding=bops["padding"].lower(),
                                      input_shape=input.shape)(input)
    op.output_layer = layer
  elif op.layer_type == "AVERAGE_POOL_2D":
    input = op.op_input[0].output_layer
    json_op = json_ops[op.output]
    bops = json_op["builtin_options"]
    layer = tf.keras.layers.AveragePooling2D(pool_size=(bops["filter_width"],
                                                        bops["filter_height"]),
                                             strides=(bops["stride_w"],bops["stride_h"]),
                                             padding=bops["padding"].lower(),
                                             input_shape=input.shape)(input)
    op.output_layer = layer
  elif op.layer_type == "RESHAPE":
    input = op.op_input[0].output_layer
    out_shape = tensors[op.output]
    layer = tf.keras.layers.Reshape(out_shape)(input)
    op.output_layer = layer
  elif op.layer_type == "SOFTMAX":
    input = op.op_input[0].output_layer
    print(input)
    layer = tf.keras.layers.Softmax()(input)
    op.output_layer = layer
  elif op.layer_type == "QUANTIZE":
    # TODO: Skipping for now
    op.output_layer = op.op_input[0].output_layer
  else:
    print("Unknown layer: " + op.layer_type + "!")
    sys.exit(1)

model = tf.keras.models.Model(inputs=a,outputs=ops[len(ops)-1].output_layer)

adam = keras.optimizers.Adam(epsilon = 1e-08)
#model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.compile()

#history = model.fit(X_full, Y_full, batch_size=128, epochs=1)
converter = tf.lite.TFLiteConverter.from_keras_model(model) #this works!!!!

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open("new_" + input_name + ".tflite", 'wb') as f:
   f.write(tflite_model)
