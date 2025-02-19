import sys
import re

names = ["ADD","AVERAGE_POOL_2D","ARG_MAX","CONCATENATION","CONV_2D","DEQUANTIZE","DEPTHWISE_CONV_2D","FULLY_CONNECTED","LOGISTIC","MAX_POOL_2D","MEAN","PAD","RESHAPE","RESIZE_BILINEAR","RESIZE_NEAREST_NEIGHBOR","QUANTIZE","SOFTMAX","TRANSPOSE_CONV","TFLite_Detection_PostProcess"]
ops = {}

for line in open("output.txt","r"):
    m = re.match("\s+Op\#\d+\s(\w+)\(.*",line)
    if m:
        op = m.group(1)
        if not op in ops:
            ops[op] = 0
        ops[op] = ops[op] + 1

output = ""

for op in ops:
    if not op in names:
        print(op)

for name in names:
    if name in ops:
        output = output + "," + str(ops[name])
    else:
        output = output + ",0"

print(output)
