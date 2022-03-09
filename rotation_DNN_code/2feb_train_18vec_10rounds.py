import tensorflow as tf
import numpy as np
import random
import time
from tensorflow import keras
from keras import backend as K
import time

def modelMaker():
    model =  keras.models.Sequential([
          #input layer
          keras.layers.InputLayer(input_shape=(18,)),
          #layer 1
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 2
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 3
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 4
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 5
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 6
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 7
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 8
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 9
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),
          #layer 10
          keras.layers.BatchNormalization(),
          keras.layers.Dense(100, activation="relu"),

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
    history = model.fit(X_train, Y_train, batch_size=128, epochs=epochs)
    toc = time.time()

    #now save everything
    np.save('history_7feb' + rot_num, history.history) #saving the history
    #to load back in: history=np.load('my_history.npy',allow_pickle='TRUE').item()
    #history in this case is a dictionary and you can find everything from the keys

    #create list of strings you need for the thingy
    L = ["\n1 degree rotated model " + str(rot_num), "\ntraintime: " + str(toc-tic), "\ntest [loss,accuracy]:" + ' '.join(map(str,model.evaluate(X_test,Y_test))), "\n"]
    #evaluate the model. In the file, first # is loss, second is acc
    #f.writelines(L)

    return model

def main():
    
    start = time.time()


    #read in original test dataset
    test = np.load("18vec_dataset/18vec_testset_round0.npy")
    X_test, Y_test = test[:,:-1], test[:,-1]

    #run it 10 times total
    acc_total = [] #final array to hold accuracy values
    for i in range(20):

        acc_round = []  #stores each round acc
        for x in range(10,101,10):

            #read in the dataset
            train = np.load("18vec_dataset/18vec_trainset_round0_rot" + str(x) + ".npy")
            np.random.shuffle(train) #shuffle the train set

            #create, train model
            model = traintest_model(train,test,str(x)+'x',epochs=100)

            #save the accuracy
            acc_round.append(model.evaluate(X_test,Y_test)[1])
            #now save it in case
            np.save("18vec_1deg_acc_round"+str(i),acc_round)

            model.save("18vec_1deg_7feb_round" + str(i) + "rotnum_" + str(x))

        acc_total.append(acc_round)

        np.save("acc_total_18vec_1deg_17feb",acc_total)
        np.savetxt("textfileacc_total_18vec_1deg_17feb",acc_total)

    end = time.time()
    print(end-start)
    f=open("timing.txt","a")
    f.write(str(end-start))

if __name__=="__main__":
    main()
