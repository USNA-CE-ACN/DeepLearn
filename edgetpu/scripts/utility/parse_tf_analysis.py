import os
import re

class Operation:
    def __init__(self):
        self.layer_type = ""
        # Tensor numbers for input, shape (Conv2D), and output
        self.number = 0
        self.op_input = []
        self.output_layer = None
        self.input_num = []
        self.filter = 0
        self.output = 0
        self.stride_w = 0
        self.stride_h = 0
        self.padding = ""
        self.activation = ""

def print_op(op):
   print(str(op.number) + ": " + op.layer_type)
   for oop in op.op_input:
       print("  " + str(oop.number))
       print("  " + str(oop.output_layer))
        
def generate_analysis_file(input_file):
    os.system("python3 analyze_model.py " + input_file + " > model.analysis")

def find_op(ops,op_num):
    for op in ops:
        if op.output == op_num:
            return op
    return None
    
def parse_analysis_file():
    ops = []
    tensors = []
    input_tensor = 0
    
    for line in open("model.analysis","r"):
        m = re.match("\s+Op\#(\d+)\s+([A-Z,a-z,0-9,\_,\-]+)\(T\#(\d+),\s(.*)\s\-\>\s\[T\#(\d+)\].*",line)
        if m:
            op = Operation()
            op.number = int(m.group(1))
            op.layer_type = m.group(2)
            op.input_num = [int(m.group(3))]
            remainder = m.group(4)
            op.output = int(m.group(5))

            if op.layer_type == "CONV_2D" or op.layer_type == "DEPTHWISE_CONV_2D" or op.layer_type == "FULLY_CONNECTED":
                m = re.match("T\#(\d+)\,.*",remainder)
                op.filter = int(m.group(1))
            elif op.layer_type == "CONCATENATION" or op.layer_type == "ADD":
                # Switch input to list of inputs           
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

            for op_num in op.input_num:
                input_op = find_op(ops,op_num)
                if input_op:
                    op.op_input.append(input_op)
                    
            ops.append(op)
        else:
            m = re.match("\s+Op\#(\d+)\s+([A-Z,a-z,0-9,\_,\-]+)\(T\#(\d+)\)\s\-\>\s\[T\#(\d+)\].*",line)
            if m:
                op = Operation()
                op.number = int(m.group(1))
                op.layer_type = m.group(2)
                op.input_num = [int(m.group(3))]
                op.output = int(m.group(4))
                for op_num in op.input_num:
                    input_op = find_op(ops,op_num)
                    if input_op:
                        op.op_input.append(input_op)
                ops.append(op)
                
            m = re.match("\s+T\#(\d+).*shape\:\[([0-9,\,,\s]+)\].*",line)
            if m:
                shape = tuple(map(int, m.group(2).split(", ")))
                tensors.append(shape)

            m = re.match(".*Subgraph\#\d+\(T\#(\d+)\)\s\-\>\s\[T\#(\d+)\].*",line)
            if m:
                input_tensor = int(m.group(1))
                output_tensor = int(m.group(2))

    return (ops,tensors,input_tensor)
