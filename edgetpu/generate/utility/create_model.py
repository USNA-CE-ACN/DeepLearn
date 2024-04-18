import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

from utility.parse_tf_analysis import *

def get_standard_conv2d(bias,filter,bops,act,input):
    layer = tf.keras.layers.Conv2D(bias,(filter[1],filter[2]),
                                   (bops["stride_w"],bops["stride_h"]),
                                   padding=bops["padding"].lower(),
                                   activation=act,
                                   use_bias=False,
                                   input_shape=input.shape)(input)
    return layer

def expand_conv2d_layer(round,input,filter,bops,act,tensors,op):
    # Single Conv2D in a column
    layer1 = get_standard_conv2d(filter[0],filter,bops,act,input)
    if round == 0:
        return layer1

    concat_list = [layer1]

    front_filter = list(filter)
    front_filter[1] = 1

    back_filter = list(filter)
    back_filter[2] = 1
    
    if round >= 1:
        # 2 Conv2Ds in a column
        layer2 = get_standard_conv2d(filter[0]/2,filter,bops,act,input)        
        layer3 = get_standard_conv2d(filter[0],filter,bops,act,layer2)
        
        if round >= 4:
            mid_concat = [layer3]
            layer4 = get_standard_conv2d(filter[0]/2,front_filter,bops,act,layer2)
            
            mid_concat.append(layer4)

            if round >= 6:
                layer5 = get_standard_conv2d(filter[0]/2,back_filter,bops,act,layer2)
                mid_concat.append(layer5)

                if round >= 8:
                    layer6 = get_standard_conv2d(filter[0],front_filter,bops,act,layer2)
                    mid_concat.append(layer6)

                    if round>= 10:
                        layer7 = get_standard_conv2d(filter[0],back_filter,bops,act,layer2)
                        mid_concat.append(layer7)
                        
            concat_list.append(tf.keras.layers.Concatenate()(mid_concat))
        else:
            concat_list.append(layer3)

    if round >= 2:
        fwidth = input.shape[1] / tensors[op.output][1]
        fheight = input.shape[2] / tensors[op.output][2]
        layer2 = tf.keras.layers.AveragePooling2D(pool_size=(fwidth,
                                                             fheight),
                                                  strides=(bops["stride_w"],bops["stride_h"]),
                                                  padding=bops["padding"].lower(),
                                                  input_shape=input.shape)(input)
        layer3 = get_standard_conv2d(filter[0],filter,bops,act,layer2)
        concat_list.append(layer3)
    if round >= 3:
        # 3 Conv2Ds in a column
        layer2 = get_standard_conv2d(filter[0]/4,filter,bops,act,input)
        layer3 = get_standard_conv2d(filter[0]/2,filter,bops,act,layer2)
        layer4 = get_standard_conv2d(filter[0],filter,bops,act,layer3)

        if round >= 5:
            mid_concat = [layer4]
            layer5 = get_standard_conv2d(filter[0]/2,front_filter,bops,act,layer3)
            mid_concat.append(layer5)

            if round >= 7:
                layer6 = get_standard_conv2d(filter[0]/2,back_filter,bops,act,layer3)
                mid_concat.append(layer6)

                if round >= 9:
                    layer7 = get_standard_conv2d(filter[0],front_filter,bops,act,layer3)
                    mid_concat.append(layer7)

                    if round>= 11:
                        layer8 = get_standard_conv2d(filter[0],back_filter,bops,act,layer3)
                        mid_concat.append(layer8)
            
            concat_list.append(tf.keras.layers.Concatenate()(mid_concat))
        else:
            concat_list.append(layer4)

    clayer = tf.keras.layers.Concatenate()(concat_list)
    # Use when condensing layers in middle
    #mplayer = tf.keras.layers.MaxPooling2D(pool_size=(1,clayer.shape[3]/layer1.shape[3]),data_format="channels_first")(clayer)
    mplayer = clayer # Use when expanding the model
    return mplayer

def create_model(ops,tensors,json_ops,model_input,round=0):   
    print("Starting create_model")
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
                layer = get_standard_conv2d(math.floor(filter[0]),filter,bops,act,input)
            else:
                layer = tf.keras.layers.DepthwiseConv2D(math.floor(filter[0]),(bops["stride_w"],bops["stride_h"]),
                                                        padding=bops["padding"].lower(),
                                                        depth_multiplier=bops["depth_multiplier"],
                                                        activation=act,
                                                        input_shape=input.shape[1:])(input)
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
                                              input_shape=input.shape[1:])(input)
            op.output_layer = layer
        elif op.layer_type == "AVERAGE_POOL_2D":
            input = op.op_input[0].output_layer
            json_op = json_ops[op.output]
            bops = json_op["builtin_options"]
            
            bops["filter_width"] = input.shape[1] / tensors[op.output][1]
            bops["filter_height"] = input.shape[2] / tensors[op.output][2]
            
            layer = tf.keras.layers.AveragePooling2D(pool_size=(bops["filter_width"],
                                                                bops["filter_height"]),
                                                     strides=(bops["stride_w"],bops["stride_h"]),
                                                     padding=bops["padding"].lower(),
                                                     input_shape=input.shape)(input)
            op.output_layer = layer
        elif op.layer_type == "RESHAPE":
            input = op.op_input[0].output_layer
            out_shape = tensors[op.output]
            layer = tf.keras.layers.Reshape((input.shape[-1],), input_shape=input.shape)(input)
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

    print("Creating model")
    model = tf.keras.models.Model(inputs=model_input,outputs=ops[len(ops)-1].output_layer)
    return model

first_input = 0
X_full = 0

def set_first_input(new_first_input):
    global first_input
    first_input = new_first_input
    global input_size
    input_size = (100,)
    
def representative_data_gen():
    X_full = np.random.rand(*(input_size + first_input))
    for i in range(3):
        yield [X_full[i].astype(np.float32)]

def output_model(model,filename):
    adam = keras.optimizers.Adam(epsilon = 1e-08)
    #model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile()
    print("Compiled model")

    #model.save(filename + ".keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model) #this works!!!!

    print("Created converter " + filename)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
#    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    print("Converted model")
    
    with open(filename, 'wb') as f:
        f.write(tflite_model)
