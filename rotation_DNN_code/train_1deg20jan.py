import tensorflow as tf
import numpy as np
import random
import time
from tensorflow import keras
from keras import backend as K


#version of rotation function that has input as flattened array
def rot(in_arr, degree, num_rot):
  """function that randomly rotates the flattened dataset (number_of_samples,121) a fixed number of times, spits out flattened array. Last value is the output class

  Parameters
  --------------
  in_arr : np.array
    dataset to rotate (number of samples,121)
    [0-119] = the flattened 3,40 array
    [120]   = the output class
  degree : float
    range of rotation. The rotation will be a random value from a uniform distribution in between +/- degree
  num_rot : int
    Factor of dataset expansion. Does (num_rot - 1) and appends to the original array, giving a dataset expansion factor of num_rot

  Returns
  ---------------
  np.array (number of samples * num_rot, 121)
    the rotated array. It is in_arr with all of the rotations appended onto the end.
    """
  #initialize flattened rotated array w arr (but flattened and results appended)
  rotated_arr = list(in_arr)
  arr, out_class = in_arr[:,:-1], in_arr[:,-1]
  arr = np.reshape(arr,(-1,3,40))
  for i in range(num_rot - 1): #rotate the dataset num amount of times
    #rotate it by +/- degree in any direction
    x, y, z = np.radians(random.random()*(2*degree) - degree), np.radians(random.random()*(2*degree) - degree), np.radians(random.random()*(2*degree) - degree) #pick a random x, y, and z
    R = np.array(((np.cos(z)*np.cos(y), np.cos(z)*np.sin(y)*np.sin(x)-np.sin(z)*np.cos(x),  np.cos(z)*np.sin(y)*np.cos(x)+np.sin(z)*np.sin(x)),
                  (np.sin(z)*np.cos(y), np.sin(z)*np.sin(y)*np.sin(x)+np.cos(z)*np.cos(x),  np.sin(z)*np.sin(y)*np.cos(x)-np.cos(z)*np.sin(x)),
                  (-np.sin(y),          np.cos(y)*np.sin(x),                                np.cos(y)*np.cos(x))))
    for j in range(len(arr)): #matrix multiplication of each (3x40) row
      rotated_arr.append(np.append(np.matmul(R,arr[j]).flatten(),(out_class[j]))) #append to the rotated array
  return np.asarray(rotated_arr)

def modelMaker():
    model =  keras.models.Sequential([
          #input layer
          keras.layers.InputLayer(input_shape=(120,)),
          #layer 1
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 2
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 3
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 4
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 5
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 6
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 7
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 8
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 9
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),
          #layer 10
          keras.layers.BatchNormalization(),
          keras.layers.Dense(1000, activation="relu"),

          #output layer
          keras.layers.Dense(9, activation="softmax")
          ])
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

def traintest_model(train, test,rot_num,epochs=200):
    keras.backend.clear_session() #clear previous session

    #separate dataset into x and y
    X_train, Y_train = train[:,:-1], train[:,-1]
    X_test, Y_test = test[:,:-1], test[:,-1]

    model = modelMaker()
    tic = time.time()
    history = model.fit(X_train, Y_train, batch_size=128, validation_split=0.2, epochs=epochs)
    toc = time.time()

    #now save everything
    np.save('history_20jan' + rot_num, history.history) #saving the history
    #to load back in: history=np.load('my_history.npy',allow_pickle='TRUE').item()
    #history in this case is a dictionary and you can find everything from the keys

    #create list of strings you need for the thingy
    L = ["\n1 degree rotated model " + str(rot_num), "\ntraintime: " + str(toc-tic), "\ntest [loss,accuracy]:" + ' '.join(map(str,model.evaluate(X_test,Y_test))), "\n"]
    #evaluate the model. In the file, first # is loss, second is acc
    #f.writelines(L)

    return model

def main():

    #load in the original dataset
    orig = np.load("acceldata_orig.npy") #continue to use this test set
    acc_total = [] #final array to hold accuracy values

    for i in range(10):

        np.random.shuffle(orig) #shuffle
        train_orig, test = orig[:9422], orig[9422:] #separate into train and test set
        X_test, Y_test = test[:,:-1], test[:,-1]
        np.save('trainset_orig_70_20jan_round' + str(i), train_orig) #save original train dataset
        np.save('testset_orig_30_20jan_round' + str(i), test) #save the original test dataset

        #array to store acc values
        acc_round = []

        for x in range(10,101,10):
            print("20jan_Round " + str(i) + " rotation num " + str(x))
            #create the rotated vector
            train = rot(train_orig, 1, x)
            model = traintest_model(train,test,str(x) + 'x', epochs=100) #run the function CHANGINGIT TO 100 JUST TO GO FASTER
            #evaluate the model and append results to end of acc[]
            acc_round.append(model.evaluate(X_test,Y_test)[1])
            #now save the accuracy just in case every time you finish training a thing
            np.save("1deg_acc_round"+str(i), acc_round)

            #save the model
            model.save('dnn_1deg_20jan_round' + str(i) + 'rotnum_' + str(x))

        acc_total.append(acc_round) #add to total accuracy

        #save final accuracy array at the end of every round
        np.save("1deg_acc_20jan", acc_total) #saving accuracy to 18 january
        np.savetxt("textfileacc",acc_total)

if __name__=="__main__":
    main()
