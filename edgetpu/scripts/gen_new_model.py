import sys,os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
import json

class Operation:
    def __init__(self):
        self.layer_type = ""
        # Tensor numbers for input, shape (Conv2D), and output
        self.number = 0
        self.op_input = []
        self.op_output = []
        self.input_num = 0
        self.filter = 0
        self.output = 0
        self.stride_w = 0
        self.stride_h = 0
        self.padding = ""
        self.activation = ""

def replace_operations(op_list, x, y, a, b, M):
    # Remove operations numbered x to y from the list
    removed_ops = op_list[x:y+1]
    for op in removed_ops:
        if op.op_input:
            op.op_input.op_output = op.op_output
        if op.op_output:
            op.op_output.op_input = op.op_input
    del op_list[x:y+1]
    
    # Generate a new list of operations to replace the removed ones
    new_ops = [op.__class__() for i in range(M) for op in op_list[a:b+1]]
    
    # Update the op_input and op_output members of the new operations
    for i in range(len(new_ops)):
        if i == 0:
            new_ops[i].op_input = removed_ops[0].op_input
        else:
            new_ops[i].op_input = new_ops[i-1]
        if i == len(new_ops) - 1:
            new_ops[i].op_output = removed_ops[-1].op_output
        else:
            new_ops[i].op_output = new_ops[i+1]

    # Insert the new operations into the original list
    if removed_ops[0].op_input:
        removed_ops[0].op_input.op_output = new_ops[0]
    if removed_ops[-1].op_output:
        removed_ops[-1].op_output.op_input = new_ops[-1]
    op_list[x:x] = new_ops
        
ops = []
tensors = []
input_tensor = 0

input_file = sys.argv[1]

os.system("python3 analyze_model.py " + input_file + " > model.analysis")
# Outputs (sys.argv[1] - .tflite) + .json in local directory
os.system("flatc -t --strict-json --defaults-json schema.fbs -- " + sys.argv[1])

for line in open("model.analysis","r"):
    m = re.match("\s+Op\#(\d+)\s+([A-Z,a-z,0-9,\_,\-]+)\(T\#(\d+),\s(.*)\s\-\>\s\[T\#(\d+)\].*",line)
    if m:
        op = Operation()
        op.number = int(m.group(1))
        op.layer_type = m.group(2)
        op.input_num = int(m.group(3))
        
        remainder = m.group(4)

        op.output = int(m.group(5))
        
        if op.layer_type == "CONV_2D" or op.layer_type == "DEPTHWISE_CONV_2D" or op.layer_type == "FULLY_CONNECTED":
            m = re.match("T\#(\d+)\,.*",remainder)
            op.filter = int(m.group(1))
        elif op.layer_type == "CONCATENATION" or op.layer_type == "ADD":
            # Switch input to list of inputs
            tmp = op.input_num
            op.input_num = [tmp]

            remainder = remainder[0:remainder.index(")")]
            
            inputs = remainder.split(",")
            for i in inputs:
                m = re.match("\s*T\#(\d+).*",i)
                if m:
                    op.input_num.append(int(m.group(1)))
        elif op.layer_type == "MEAN":
            m = re.match("\s*T\#(\d+)\[(\d+)\,\s(\d+)\].*",remainder)
            if m:
                filter = m.group(1)
            else:
                print("Failed to parse MEAN!")                    
                    
        ops.append(op)
    else:
        m = re.match("\s+Op\#(\d+)\s+([A-Z,a-z,0-9,\_,\-]+)\(T\#(\d+)\)\s\-\>\s\[T\#(\d+)\].*",line)
        if m:
            op = Operation()
            op.number = int(m.group(1))
            op.layer_type = m.group(2)
            op.input_num = int(m.group(3))
            op.output = int(m.group(4))
            ops.append(op)

    m = re.match("\s+T\#(\d+).*shape\:\[([0-9,\,,\s]+)\].*",line)
    if m:
        shape = tuple(map(int, m.group(2).split(", ")))
        tensors.append(shape)

    m = re.match(".*Subgraph\#\d+\(T\#(\d+)\)\s\-\>\s\[T\#(\d+)\].*",line)
    if m:
        input_tensor = int(m.group(1))
        output_tensor = int(m.group(2))

json_ops = {}
        
# Parse JSON to find
input_name = input_file[input_file.rindex("/")+1:input_file.rindex(".")]
input_json = input_name + ".json"
with open(input_json) as f:
    model_json = json.load(f)
    sg_ops = model_json["subgraphs"][0]["operators"]
    for sg_op in sg_ops:
        # Look up json based on output number to find the rest
        json_ops[sg_op["outputs"][0]] = sg_op

# TODO: Fix this to use input size from the model!
print(input_tensor)
input = tensors[input_tensor]
print(input)

input_size = (100,)
X_full = np.random.rand(*(input_size + input))
rng = np.random.default_rng()
Y_full = rng.integers(0,1000,100)

def representative_data_gen():
  for i in range(100):
    yield [X_full[i].astype(np.float32)]

keras.backend.clear_session()

a = tf.keras.layers.Input(shape=input[1:])

for onum in range(0,len(ops)):
    print(str(onum) + ": " + op.layer_type)

firstLayerRep = int(input("\nEnter the number of the first layer you want to replicate"))
lastLayerRep = int(input("Enter the number of the last layer you want to replicate (same number as first if repeating single layer"))

firstLayerRemove = int(input("Enter the number of the first layer you want to remove"))
lastLayerRemove = int(input("Enter the number of the last layer you want to remove"))

# First, extract the layers that we'll want to repeat
repeat_ops = []

for onum in range(int(firstLayerRep),int(lastLayerRep)+1):
    repeat_ops.append(ops[onum])

# Remove the existing layers
del ops[int(firstLayerRemove):int(lastLayerRemove)]

# Now, clone layers at the insertion point and generate models
# TODO

layers = {}
layers[ops[0].input_num] = a

for op in ops:
    if op.layer_type == "CONV_2D" or op.layer_type == "DEPTHWISE_CONV_2D":
        input = layers[op.input_num]
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
        layers[op.output] = layer
    elif op.layer_type == "FULLY_CONNECTED":
        input = layers[op.input_num]
        dim = filter[0]
        json_op = json_ops[op.output]
        bops = json_op["builtin_options"]
        act = None
        if bops["fused_activation_function"] != "NONE":
            act = bops["fused_activation_function"].lower()
        layer = tf.keras.layers.Dense(dim,act)(input)
        layers[op.output] = layer
    elif op.layer_type == "CONCATENATION":
        concat_list = []
        for input in op.input_num:
            concat_list.append(layers[input])
        layer = tf.keras.layers.Concatenate()(concat_list)
        layers[op.output] = layer
    elif op.layer_type == "ADD":
        add_list = []
        for input in op.input_num:
            add_list.append(layers[input])
        layer = tf.keras.layers.Add()(add_list)
        layers[op.output] = layer
    elif op.layer_type == "MEAN":
        input = layers[op.input_num]
        layer = tf.keras.layers.GlobalAveragePooling2D()(input)
        layers[op.output] = layer
    elif op.layer_type == "MAX_POOL_2D":
        input = layers[op.input_num]
        json_op = json_ops[op.output]
        bops = json_op["builtin_options"]
        layer = tf.keras.layers.MaxPool2D((bops["filter_width"],bops["filter_height"]),
                                           (bops["stride_w"],bops["stride_h"]),
                                           padding=bops["padding"].lower(),
                                           input_shape=input.shape)(input)
        layers[op.output] = layer
    elif op.layer_type == "AVERAGE_POOL_2D":
        input = layers[op.input_num]
        json_op = json_ops[op.output]
        bops = json_op["builtin_options"]
        layer = tf.keras.layers.AveragePooling2D(pool_size=(bops["filter_width"],
                                                            bops["filter_height"]),
                                                 strides=(bops["stride_w"],bops["stride_h"]),
                                                 padding=bops["padding"].lower(),
                                                 input_shape=input.shape)(input)
        layers[op.output] = layer
    elif op.layer_type == "RESHAPE":
        input = layers[op.input_num]
        out_shape = tensors[op.output]
        layer = tf.keras.layers.Reshape(out_shape)(input)
        layers[op.output] = layer
    elif op.layer_type == "SOFTMAX":
        input = layers[op.input_num]
        layer = tf.keras.layers.Softmax()(input)
        layers[op.output] = layer
    elif op.layer_type == "QUANTIZE":
        layers[op.output] = layers[op.input_num]
    else:
        print("Unknown layer: " + op.layer_type + "!")
        sys.exit(1)

model = tf.keras.models.Model(inputs=a,outputs=layers[ops[len(ops)-1].output])

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
