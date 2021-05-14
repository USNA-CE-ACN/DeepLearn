#by Phil Ross
#Additions to both versions of code - add an inference timer to see the speedup from CPU vs TPU
#possible useful library inclusions-- pycoral.utils, .adapters
#import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify
import os
import pathlib
import numpy as np
import time

# Specify the TensorFlow model, labels,inputs and classifiers
class_names = ["Cont Walk", "Foraging", "Other", "Resting", "Rooting", "Running", "Stanging", "Trotting", "Vigilance"]
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'dnn_baseline_nonorm_quant_edgetpu.tflite')
input_file = os.path.join(script_dir, 'input_array.npy')
output_file = os.path.join(script_dir, 'output_array.npy')
input_np = np.load(input_file)
output_np = np.load(output_file)
output_np = np.array(output_np).astype(int)


def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    scale, zero_point = input_details['quantization']
    #WIP


#Build Interpreter and allocate tensors
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()
print("Interpreter:")
print(interpreter)

#Get input & output details
input_details = interpreter.get_input_details()[0]
print("Input Details:")
print(input_details)
tensor_index = input_details['index']
input_tensor = interpreter.tensor(tensor_index)()[0]
scale, zero_point = input_details['quantization']
output_details = interpreter.get_output_details()

#Set Accuracy and Timing Variables
numer = 0
dem = 13191
invoke_time_total = 0

#initialize data
input_data = np.array(input_np, dtype = np.float32)

#this test rotates through the same piece of data continously. Ideally there should be significant speedup
print('Beginning Inferencing...')
start = time.perf_counter()
for i in range(13191):
    input_data_new = input_data[i]
    input_tensor[:] = np.uint8(input_data_new / scale + zero_point)
    intense = common.input_tensor(interpreter)
    #print("tensor")
    #print(intense)
    #print("data")
    #print(input_data_new)
    invoke_time_start = time.perf_counter()
    set_input_tensor(interpreter, input_data_new)

    invoke_time = time.perf_counter() - invoke_time_start
    invoke_time_total+=invoke_time
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = scale * (output_data - zero_point)
    #print("Output data:")
    #print(output_data)
    #print("Prediction: " + class_names[np.argmax(output_data)])
    #print("Actual: " +class_names[output_np[i]])
    if class_names[np.argmax(output_data)] == class_names[output_np[i]]:
        numer += 1

#Get total time of loop and print out results
inference_time = time.perf_counter()-start
print('------RESULTS------')
print('Total loop Time: %.1fms' % (inference_time *1000))
print('Invoke Time: %.1fms' % (invoke_time_total *1000))
print("Accuracy: " + str(numer*100/dem) + "%")
