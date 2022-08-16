"""
This is for the CNN 1 deg raw data
- new model structure
"""

import tensorflow as tf
import numpy as np
import random
import time
from keras import datasets, layers, models
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def get_model(input_shape = (40,3), num_classes = 9, learning_rate = 1e-4, l2_rate = 1e-4, kernel_size = 1, strides = 1):

  model = models.Sequential()


  model.add(Conv1D(
      64, kernel_size = kernel_size, #kernel size = 1 means we're looking at 1 row at a time (so 1x3)
			strides = strides,
			padding = "valid",
			input_shape=input_shape)
  )
  model.add(MaxPooling1D(pool_size=5, padding='same'))
  model.add(BatchNormalization())
  model.add(layers.ReLU())

  model.add(Conv1D(
      64, kernel_size = 3
  ))
  model.add(MaxPooling1D(pool_size=4, padding='same'))
  model.add(BatchNormalization())
  model.add(layers.ReLU())

  model.add(Flatten())

  model.add(Dense(9,activation='softmax'))

  model.compile(optimizer = 'adam',
				loss = "sparse_categorical_crossentropy",
        metrics = ['accuracy'])
				#metrics = ["categorical_accuracy"])
  return model

def main():

      test_full = np.load("../dataset/testset.npy")
      X_test, Y_test = test_full[:,:-1], test_full[:,-1]
      #reshape X to be (num samples, 40, 3)
      X_test = X_test.reshape(X_test.shape[0],3,40)
      X_test = X_test.swapaxes(1,2)
      acc_total = []
      time_total = []
      for i in range(25):
          acc_round = []
          time_round = []
          for x in range(0,101,10):
              #read in the training data and split it
              train_full = np.load("../dataset/trainset_"+str(x)+"x_1deg.npy")
              np.random.shuffle(train_full) #shuffle dataset
              X_train, Y_train = train_full[:,:-1], train_full[:,-1]
              X_train = X_train.reshape(X_train.shape[0],3,40).swapaxes(1,2) #reshape to 40,3
              model = get_model() #make the model

              #train the model and time it
              tic = time.time()
              #try earlystopping
              history = model.fit(X_train,Y_train, validation_split = .2,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True,patience=3)],
                                  batch_size=128,epochs=50)
              toc = time.time()
              time_round.append(toc-tic) #save timing data for that model

              #save the history and time for that round as well as the model
              np.save("timeCNN_round"+str(i),time_round)
              np.save('historyCNNraw_'+str(x)+"x",history.history) #save history
              model.save('CNNraw_'+str(x)+"x") #save the model
              #save accuracy
              acc_round.append(model.evaluate(X_test,Y_test)[1])
              np.save("accCNN_round"+str(i),acc_round)
              acc_total.append(acc_round)
              time_total.append(time_round)

      np.save("acc_total_CNNraw",acc_total)
      np.save("time_total_CNNraw",time_total)

if __name__=='__main__':
    main()
