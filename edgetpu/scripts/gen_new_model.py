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
        self.input = 0
        self.filter = 0
        self.output = 0
        self.stride_w = 0
        self.stride_h = 0
        self.padding = ""
        self.activation = ""
        
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
        op.input = int(m.group(3))
        
        remainder = m.group(4)

        op.output = int(m.group(5))
        
        if op.layer_type == "CONV_2D" or op.layer_type == "DEPTHWISE_CONV_2D" or op.layer_type == "FULLY_CONNECTED":
            m = re.match("T\#(\d+)\,.*",remainder)
            op.filter = int(m.group(1))
        elif op.layer_type == "CONCATENATION" or op.layer_type == "ADD":
            # Switch input to list of inputs
            tmp = op.input
            op.input = [tmp]

            remainder = remainder[0:remainder.index(")")]
            
            inputs = remainder.split(",")
            for i in inputs:
                m = re.match("\s*T\#(\d+).*",i)
                if m:
                    op.input.append(int(m.group(1)))
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
            op.input = int(m.group(3))
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

layers = {}
layers[ops[0].input] = a

for op in ops:
    if op.layer_type == "CONV_2D" or op.layer_type == "DEPTHWISE_CONV_2D":
        input = layers[op.input]
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
        input = layers[op.input]
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
        for input in op.input:
            concat_list.append(layers[input])
        layer = tf.keras.layers.Concatenate()(concat_list)
        layers[op.output] = layer
    elif op.layer_type == "ADD":
        add_list = []
        for input in op.input:
            add_list.append(layers[input])
        layer = tf.keras.layers.Add()(add_list)
        layers[op.output] = layer
    elif op.layer_type == "MEAN":
        input = layers[op.input]
        layer = tf.keras.layers.GlobalAveragePooling2D()(input)
        layers[op.output] = layer
    elif op.layer_type == "MAX_POOL_2D":
        input = layers[op.input]
        json_op = json_ops[op.output]
        bops = json_op["builtin_options"]
        layer = tf.keras.layers.MaxPool2D((bops["filter_width"],bops["filter_height"]),
                                           (bops["stride_w"],bops["stride_h"]),
                                           padding=bops["padding"].lower(),
                                           input_shape=input.shape)(input)
        layers[op.output] = layer
    elif op.layer_type == "AVERAGE_POOL_2D":
        input = layers[op.input]
        json_op = json_ops[op.output]
        bops = json_op["builtin_options"]
        layer = tf.keras.layers.AveragePooling2D(pool_size=(bops["filter_width"],
                                                            bops["filter_height"]),
                                                 strides=(bops["stride_w"],bops["stride_h"]),
                                                 padding=bops["padding"].lower(),
                                                 input_shape=input.shape)(input)
        layers[op.output] = layer
    elif op.layer_type == "RESHAPE":
        input = layers[op.input]
        out_shape = tensors[op.output]
        layer = tf.keras.layers.Reshape(out_shape)(input)
        layers[op.output] = layer
    elif op.layer_type == "SOFTMAX":
        input = layers[op.input]
        layer = tf.keras.layers.Softmax()(input)
        layers[op.output] = layer
    elif op.layer_type == "QUANTIZE":
        layers[op.output] = layers[op.input]
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
