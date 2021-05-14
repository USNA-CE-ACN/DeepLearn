//By Philip Ross. This is a running model for a c++ version of the boar_run.py code.
//Ideally this code will allow for speedup in both the on computer and embedded versions
//of this code

#include <cstdio>
#include <ctime>
#include <NumCpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define TFLITE_MINIMAL_CHECK(x)
  if(!(x)){
    fprint(stderr, "error at %s:%d\n", __FILE__, __LINE__);
    exit(1);
  }

int main(){

  //class names for different boar behavior
  std::string *class_names = new std::string[9];
  class_names[0] = "Cont Walk";
  class_names[1] = "Foraging";
  class_names[2] = "Other";
  class_names[3] = "Resting";
  class_names[4] = "Rooting";
  class_names[5] = "Running";
  class_names[6] = "Standing";
  class_names[7] = "Trotting";
  class_names[8] = "Vigilance";

  //Load the inputs and check data
  std::string file_input  = "input_array.npy";
  std::string file_output = "output_array.npy";
  nc::NdArray<float> input_np = nc::load<float>(file_input);
  nc::NdArray<int> output_np = nc::load<int>(file_output);


  //load in the model
  std::string filename = "boar_model_basic.tflite";
  std::unique_ptr<tflite::FlatBufferModel> model =
  tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InderpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);


  //allocate tensor buffers
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() ==kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());


  //Fill input buffers
  //TODO

  // Resize input tensors, if desired.
  interpreter->AllocateTensors();
  
  for(int i =0; i<13191; i++){
    float* input = interpreter->typed_input_tensor<float>(0);
    // Fill `input`.
    interpreter->Invoke();
    float* output = interpreter->typed_output_tensor<float>(0);
  }

  return 0;
}
