boar_run.py and boar_analysis2.py README.txt File

By Philip Ross
15 APR 2021

The codes are for boar_run.py and boar_analysis.py are both extremely straight forward.
They take in no arguments and directly grab the model file, input file, and output file
from the OS, and the classifiers for the model are few and therefore built directly into the 
code.

The code takes the boar_model_basic.tflite file, by Midshipman Jennifer Jung, as well as input
accelerometer data(formatted as a 1x8 vector of floats...I'm not sure how), and predicts the 
behavior of these boars in one of eight categories(ie walking, foraging, etc.). The code then
compares these results against output_array.npy, which has the correct answers, to figure out the 
accuracy of the model. 

Therefore, it is important that the names of those three files stay the same, they are:

Model: 
Input: input_array.npy
Output output_array.npy

In order to change which input arrays, output arrays, or models you would like to use,
you need to go in and change the .py codes.

Before you run this code, you'll need a few things:

---FOR BOTH---
Download all the files(duh), and place them in the same directory(not actually necessary, but why risk it?)

---FOR THE boar_analysis.py---
1. an Edge TPU development board by google. You can read about the device, as well as how to set it up here:
https://coral.ai/docs/dev-board/get-started/

	This website goes through all the steps needed to setup the Dev board for a linux machine. If your using
an academy machine, you won't be able to use it on every computer. Unless the dev board is being used on a computer 
it recognizes, it won't connect. It needs the first machine it connected with to add a new key from the machine to 
connect. Talk with Professor Delozier, ECE Dept USNA, if you are having this problem.

You will also need to push boar_analysis.py, input_array.npy, and output_array.npy to the device. That can be done using

	mdt push /complete/path/to/file/boar_analysis.py
	mdt push /complete/path/to/file/input_array.npy
	mdt push /complete/path/to/file/output_array.npy

I would recommend making a directory to hold all these files.

---FOR boar_run.py---
1. you will need to download: 

	tensorflow
	python3 		

downloads aren't trivial at the naval academy, hence you probably should do this work on, a. your own personal machine, or b. a dept setup box.

2. You will also need to compile any models you use using the TFLite compiler.
	To Install, run the following commands:

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt-get update

sudo apt-get install edgetpu-compiler

3. From there, run:

edgetpu_compiler -s input_filename.tflite

	this step will allow you to develop a model which utilizes the EdgeTPU hardware


Once you've taken all the necessary steps in setup, to run the code, simply enter the terminal and type

python3 boar_run.py

---OR if you have a coral accelerator/are running a shell on a Accelerator DEV Board---

python3 boar_analysis.py

Make sure all these files are in the same directory before running.

On the back end of this code, you can expect it to output:

1. Some errors if your computer doesn't have a GPU, ignore them

2. Debugging printouts, also can be ignored for most purposes
  
3. the Results- this is what we care about, and there are 3 parts:
	a. the overall time it took to loop through the code,
	   this includes the for loop of all the inferencing,
	   as well as loading the models and resizing the data.
	b. the inferencing time, this timer only counts how long
	   all of the tensor.invoke() functions took, which is the 
	   actual running of the neural net model(and loading of
	   the model if that was the first time it was set)
	c. The accuracy, ie how many predictions from the model were
	   right according to the output array.

If you have any questions about this code, please send them to my github account,

philipross2017 
