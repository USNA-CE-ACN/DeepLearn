import sys
import tflite_runtime.interpreter as tflite
from tensorflow.lite.python.util import convert_bytes_to_c_source

interpreter = tflite.Interpreter(model_path=sys.argv[1])

source_text, header_text = convert_bytes_to_c_source(tflite_model,  "model")

with open('model.h',  'w') as file:
    file.write(header_text)

with open('model.cc',  'w') as file:
    file.write(source_text)
