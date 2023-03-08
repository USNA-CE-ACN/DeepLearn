import sys,os
import subprocess
import re

CPU_FLOPS = 64697 * 1000000
input_file = "/home/ladmin/DeepLearn/models/CPU/inception/inception_v1_224_quant.tflite"

os.system("python3 analyze_model.py " + input_file + " > model.analysis")

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

output = subprocess.check_output(["/home/ladmin/Downloads/linux_x86-64_benchmark_model", "--enable_op_profiling=true",
                                  "--graph=" + input_file],stderr=subprocess.STDOUT)

start = False

flops = 0

op_flop = []
current = 0
next_inputs = []

op_map = {}
current = 0

for op in ops:
    op_map[op.input] = op
    current = current + 1

parallel = [ops[0].input]
processed = []

while len(parallel) > 0:
    next_parallel = []
    print(parallel)
    for i in parallel:
        processed.append(i)

    for op in ops:
        if op.output in processed:
            continue
        if type(op.input) == list:
            doAdd = True
            for inp in op.input:
                if not inp in processed:
                    doAdd = False
            if doAdd:
                next_parallel.append(op.output)
                    
        elif op.input in processed:
            next_parallel.append(op.output)

    parallel = next_parallel

for line in output.split('\n'):
    m = re.match(".*Operator-wise Profiling Info for Regular Benchmark Runs.*",line)
    if m:
        start = True

    m = re.match(".*Top by Computation Time.*",line)
    if m:
        start = False
        
    if start:
        m = re.match(".*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)%\s*(\d+\.\d+)%\s*(\d+\.\d+)\s*(\d+).*",line)
        if m:
            flops = flops + (float(m.group(2)) * float(m.group(6)) * CPU_FLOPS)

print(str(len(ops)) + ","  + str(flops))
        
sys.exit(1)

# ~/Downloads/linux_x86-64_benchmark_model --enable_op_profiling=true --graph=../../models/CPU/inception/inception_v1_224_quant.tflite
script = "model_operations.py"
path = '../../models/CPU/'

for (path, dirs, files) in os.walk(path):
    for f in files:
        os.system("python3 " + script + " " + path + f)
