

import tensorflow as tf
import numpy as np
import os
import cv2
from scipy import misc, stats
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import math
import random
from random import randint
from tensorflow import image


for modelnum in range (10):
    tf.reset_default_graph()

    # Starting Parameters: Size and NumOfChannels is changing in the code!
    BatchLength = 20
    NumIteration = 200001;
    LearningRate = 1e-3# learning rate of the algorithm
    NumClasses = 2 # number of output classes
    EvalFreq = 100  # evaluate on every 100th iteration
    NumOfChannels=[3,64,128] #number of channels/layer including the input picture (3:RGB,4:RGBA)
    Size=[32,32,3]
    ImSize=(Size[1],Size[0])
    PatchSize=[[5,5],[3,3]]
    AbSize=[32,32,1]
    Stride=1
    Pooling=2
    map='10_abs_models'
    imgMode='RGB'
    Abmode='F'
    ar_len=100000
    factor=4

    #placeholders

    TrainInput = tf.placeholder(tf.float32, [None, Size[0], Size[1],Size[2] ],name='TrainInput') #network input
    LabelOutput = tf.placeholder(tf.int64, [None],name='DesiredLabelOutput') #desired network output
    ExpectedOutput = tf.placeholder(tf.float32, [None, AbSize[0], AbSize[1],AbSize[2] ],name='ExpectedOutput') #desired network output

    learningRatePlaceHolder=tf.placeholder(tf.float32,name="LearningRateHolder")

    #############################################################LOADING DATA


    (TrainImages,TrainLabels),(TestImages,TestLabels)=tf.keras.datasets.cifar10.load_data()
    AbstractIm=np.zeros([NumClasses,AbSize[0], AbSize[1], AbSize[2]])
    LossOut=np.zeros([BatchLength,Size[0], Size[1], Size[2]])
    ExpOut=np.zeros([BatchLength,AbSize[0], AbSize[1], AbSize[2]])

class ShiftRegister():
    def __init__(self,regsize):
        self.Registers = np.ones([regsize], dtype=np.float32)*200
    def shifting(self):
        for i in range(0,len(self.Registers)-1):
            self.Registers[i] = self.Registers[i+1]
    def put(self,x):
        self.shifting()
        self.Registers[-1]=x

    def overfitted(self):
        return (all(self.Registers[-1] >= i for i in self.Registers)) and not (all(self.Registers[-1] == i for i in self.Registers))


    shiftregister=ShiftRegister(10)

    k=0
    for b,label in enumerate(TrainLabels):
        if label==0.0:
            TrainLabels[k]=0
            image=TrainImages[b]
            TrainImages[k]=image
            k+=1
        if label==7.0:
            TrainLabels[k]=1
            image=TrainImages[b]
            TrainImages[k]=image
            k+=1
    TrainImages=TrainImages[0:k,:,:,:]
    TrainLabels=TrainLabels[0:k]


    j=0
    for b,label in enumerate(TestLabels):
        if label==0.0:
            TestLabels[j]=0
            image=TestImages[b]
            TestImages[j]=image
            j+=1
        if label==7.0:
            TestLabels[j]=1
            image=TestImages[b]
            TestImages[j]=image
            j+=1
    TestImages=TestImages[0:j,:,:,:]
    TestLabels=TestLabels[0:j]


    path='./abstract_v2/'
    file_set=np.array(os.listdir(path))
    abcount=0
    for file_name in file_set:
        Im=misc.imread(path+str(file_name),mode=Abmode)
        Im = cv2.resize(Im,ImSize,interpolation=cv2.INTER_AREA)
        AbstractIm[abcount] = Im.reshape(AbSize)
        abcount+=1




    ##########################################FUNCTIONS



    def LeakyReLU(Input):
        # leaky ReLU
        alpha = 0.001
        Output = tf.maximum(alpha * Input, Input)
        return Output

    def shuffle(data,labels):
        num=len(data)
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def next_batch(num, data, labels):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]

        data_shuffle = np.zeros([num, int(Size[0]), int(Size[1]), int(Size[2])])
        labels_shuffle = np.zeros([num])
        abs_shuffle = np.zeros([num, int(AbSize[0]), int(AbSize[1]), int(AbSize[2])])

        maskSize = [96, 96, 1]
        maskedcounter = 0

        for i in idx:
            data_ = np.copy(data[i])
            label_ = np.copy(labels[i])
            label = int(label_)
            abstract_ = np.copy(AbstractIm[label])

            data_shuffle[maskedcounter] = data_
            abs_shuffle[maskedcounter] = abstract_
            labels_shuffle[maskedcounter] = label_
            maskedcounter += 1

        return data_shuffle, labels_shuffle, abs_shuffle


    def p_s(num):
        return PatchSize[num][0],PatchSize[num][1]


    #batchnorm conv with leakyrelu
    def bn_conv(LayerNum,xInput,nametag):
        with tf.variable_scope('conv' + str(LayerNum)+nametag):
            p1,p2=p_s(LayerNum-1)
            W=tf.get_variable('W',[p1,p2,NumOfChannels[LayerNum-1],NumOfChannels[LayerNum]])
            Y=tf.nn.conv2d(xInput,W,strides=[1, Stride, Stride, 1], padding='SAME')

            #batch norm
            beta = tf.get_variable('beta', [NumOfChannels[LayerNum]], initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', [NumOfChannels[LayerNum]], initializer=tf.constant_initializer(1.0))
            Mean, Variance = tf.nn.moments(Y, [0, 1, 2])
            Y_bn = tf.nn.batch_normalization(Y, Mean, Variance, beta, gamma, 1e-10)


            Y_sigmo=tf.sigmoid(Y_bn)
        return Y_sigmo

    def bn_conv_inp(LayerNum,xInput,nametag):
        with tf.variable_scope('conv' + str(LayerNum)+nametag):
            p1,p2=p_s(LayerNum-1)
            W=tf.get_variable('W',[p1,p2,NumOfChannels[LayerNum-1],NumOfChannels[LayerNum]])
            Y=tf.nn.conv2d(xInput,W,strides=[1, Stride, Stride, 1], padding='SAME')

            #batch norm
            beta = tf.get_variable('beta', [NumOfChannels[LayerNum]], initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', [NumOfChannels[LayerNum]], initializer=tf.constant_initializer(1.0))
            Mean, Variance = tf.nn.moments(Y, [0, 1, 2])
            Y_bn = tf.nn.batch_normalization(Y, Mean, Variance, beta, gamma, 1e-10)

        return Y_bn


    def resize_conv(LayerNum,Input):
        with tf.variable_scope('resize_conv'+str(LayerNum)):
            p1, p2 = p_s(LayerNum-1)
            W = tf.get_variable('W', [p1, p2, NumOfChannels2[LayerNum-1], NumOfChannels2[LayerNum]])

            dim = [int(Size[0] * 2), int(Size[1] * 2)]
            tf.map_fn(lambda x: tf.image.resize_images(x, dim , method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), Input)

            Y = tf.nn.conv2d(Input, W,strides= [1, 1,1, 1], padding='SAME')

            # batch norm
            beta = tf.get_variable('beta', [NumOfChannels2[LayerNum]], initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', [NumOfChannels2[LayerNum]], initializer=tf.constant_initializer(1.0))
            Mean, Variance = tf.nn.moments(Y, [0, 1, 2])
            Y_bn = tf.nn.batch_normalization(Y, Mean, Variance, beta, gamma, 1e-10)


            Size[0] = Size[0] * 2
            Size[1] = Size[1] * 2

            Y_sigmo=tf.sigmoid(Y_bn)

        return Y_sigmo


    def bn_conv2(LayerNum,xInput,nametag):
        with tf.variable_scope('conv' + str(LayerNum)+nametag):
            p1,p2=p_s(LayerNum-1)
            W=tf.get_variable('W',[p1,p2,NumOfChannels2[LayerNum],NumOfChannels2[LayerNum]])
            Y=tf.nn.conv2d(xInput,W,strides=[1, Stride, Stride, 1], padding='SAME')

            #batch norm
            beta = tf.get_variable('beta', [NumOfChannels2[LayerNum]], initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', [NumOfChannels2[LayerNum]], initializer=tf.constant_initializer(1.0))
            Mean, Variance = tf.nn.moments(Y, [0, 1, 2])
            Y_bn = tf.nn.batch_normalization(Y, Mean, Variance, beta, gamma, 1e-10)

            Y_sigmo = tf.sigmoid(Y_bn)

        return Y_sigmo

    #pooling
    def pooling(Y):
        Y = tf.nn.max_pool(Y, ksize=[1, Pooling, Pooling, 1], strides=[1, Pooling,Pooling, 1], padding='SAME')
        Size[0] = Size[0] / 2
        Size[1] = Size[1] / 2
        return Y

    ####################Data process

    TestImages,TestLabels=shuffle(TestImages,TestLabels)
    TrainImages,TrainLabels=shuffle(TrainImages,TrainLabels)



    # BOTTOM LAYER BELONGS TO THE LEFT SIDE, OUTPUT GENERATING LAYER BELONGS TO THE RIGHT SIDE!
    #Left Side
    LeftLayer1=bn_conv_inp(1,TrainInput,'l1')
    LeftLayer1_p=pooling(LeftLayer1)

    #Bottom Layer (Belongs to the left side, therefore LayerNum last LayerNum at the left side+1)
    BottomLayer=bn_conv(2,LeftLayer1_p,'b')

    NumOfChannels2=np.flip(NumOfChannels[1:],0)

    #Right Side: ascending order from bottom to top!!!

    RightLayer1=resize_conv(1,BottomLayer)
    RightLayer1=bn_conv2(1,RightLayer1,'r1')


    dim = [int(Size[0]), int(Size[1])]
    LastInput=tf.map_fn(lambda x: tf.image.resize_images(x, dim , method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), RightLayer1)


    #Last, Outout generating layer
    W_last=tf.get_variable('W_last',[PatchSize[0][0],PatchSize[0][0],NumOfChannels2[-1],1])
    B_last=tf.zeros({1})
    Y_last=tf.nn.conv2d(LastInput,W_last,strides=[1,1,1,1],padding='SAME')
    Y_last=tf.add(Y_last,B_last)
    Y_last=tf.nn.relu(Y_last,name='Output')

    Output=Y_last

    # Loss calculation (abstract picture vs. Output picture)
    with tf.name_scope('loss'):
        # L1 loss
        AbsDif = tf.abs(tf.subtract(ExpectedOutput, Output))
        shape=AbsDif.shape
        # this part implements soft L1
        Comp = tf.constant(np.ones([shape[1],shape[2],shape[3]]), dtype=tf.float32)
        SmallerThanOne = tf.cast(tf.greater(Comp, AbsDif), tf.float32)
        LargerThanOne = tf.cast(tf.greater(AbsDif, Comp), tf.float32)
        ValuestoKeep = tf.subtract(AbsDif, tf.multiply(SmallerThanOne, AbsDif))
        ValuestoSquare = tf.subtract(AbsDif, tf.multiply(LargerThanOne, AbsDif))
        Loss = tf.add(ValuestoKeep, tf.square(ValuestoSquare))

        # average loss
        loss = tf.sqrt(tf.reduce_mean(Loss),name='Loss')

    # Use ADAM optimizer this is currently the best performing training algorithm in most cases
    train_step = tf.train.AdamOptimizer(learningRatePlaceHolder).minimize(Loss)

    with tf.name_scope('stoploss'):
        #
        Exp255=tf.multiply(tf.cast(tf.greater(ExpectedOutput,170), tf.float32),255)
        Out255=tf.multiply(tf.cast(tf.greater(Output,170), tf.float32),255)
        AbsDif = tf.abs(tf.subtract(Exp255, Out255))
        StopLoss = AbsDif

        # average loss
        stoploss = tf.sqrt(tf.reduce_mean(StopLoss),name='stopLoss')



    #Accuracy: Finding the label from the output picture, calculating classifying accuracy

    with tf.name_scope('accuracy'):
        IdealOutputs = tf.constant(AbstractIm, tf.float32)  # 10,28,28,1 az mnist_refs tombben vannak az abstract kimenetek
        IdealOutputs = tf.expand_dims(IdealOutputs, 0)  # 1,10,28,28,1
        IdealOutputs = tf.tile(IdealOutputs, [BatchLength, 1, 1, 1,1])  # 16,10,28,28,1
        OutputForEveryClass = tf.expand_dims(Output,1)  # 16,1,28,28,1
        OutputForEveryClass = tf.tile(OutputForEveryClass, [1, NumClasses, 1, 1,1])  # 16,10,28,28,1
        Diffs = tf.subtract(IdealOutputs, OutputForEveryClass)
        Diffs = tf.square(Diffs)
        Diffs = tf.reduce_mean(Diffs, [2, 3, 4],name='AccLoss')
        WinnerIndices = tf.argmin(Diffs, 1,name='PredictedLabel')
        CorrectPredictions = tf.equal(WinnerIndices, LabelOutput)
        Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32),name='Accuracy')

    #Save results to csv file

    resultsFile=open("./accuracies_abs_model/accuracy_"+str(modelnum)+".csv",'w')
    resultsFile.write("Iteration;Training Accuracy;Training Loss;Test Accuracy;Test Loss")
    resultsFile.write("\n")



    ####### MODEL #######

    init=tf.global_variables_initializer()
    locinit=tf.local_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    sess.run(locinit)

    #saving out the network
    saver = tf.train.Saver(max_to_keep=None)


    for i in range(1,NumIteration):
        batch_X, batch_Y,batch_abs = next_batch(BatchLength,TrainImages,TrainLabels)
        if i%200==0:
            batch_X_t,batch_Y_t,batch_abs_t = next_batch(BatchLength, TestImages,TestLabels)
            l_tr,o_tr,a_tr= sess.run([loss,Output,Accuracy], {TrainInput: batch_X, LabelOutput: batch_Y, ExpectedOutput: batch_abs,learningRatePlaceHolder:LearningRate})
            l_test,o_test,a_test=sess.run([loss,Output,Accuracy],{TrainInput: batch_X_t,LabelOutput: batch_Y_t, ExpectedOutput: batch_abs_t,learningRatePlaceHolder:LearningRate})
            print(str(i)+' Train loss: '+str(l_tr)+' Train Accuracy: '+str(a_tr)+' Test loss: '+str(l_test)+' Test Accuracy: '+str(a_test))
            resultsFile.write(str(i) + ";" + str(a_tr) + ";" +str(l_tr)+ ";" + str(a_test)+";"+str(l_test))
            resultsFile.write("\n")
            resultsFile.flush()
        if i==100000:
            saver.save(sess, "10_model_abs/mymodel_" + str(modelnum), i)

        sess.run(train_step, {TrainInput: batch_X, LabelOutput: batch_Y, ExpectedOutput: batch_abs,learningRatePlaceHolder:LearningRate})
        if i==50000:
            LearningRate=1e-4
            print('lelassulok picit')
        if i==100000:
            LearningRate=1e-5
            print('lelassulok még egy picit')

    # independent test accuracy
    TotalAcc = 0.0
    Iter = 0
    print(type(BatchLength))
    Size=np.array(Size).astype(np.int32)
    Data =np.zeros([BatchLength] + Size)
    for i in range(0, TestImages.shape[0] - BatchLength, BatchLength):
        Data = TestImages[i:(i + BatchLength), :, :, :]
        Label = TestLabels[i:(i + BatchLength)]
        Label = np.reshape(Label, (BatchLength))
        acc = sess.run(Accuracy, feed_dict={TrainInput: Data, LabelOutput: Label})
        TotalAcc += acc
        Iter += 1
        accontest=float(TotalAcc) / float(Iter)
    print("Independent Test set: " + str(float(TotalAcc) / float(Iter)))
    np.save('./model_acc_abs/accuracy_'+str(modelnum)+'.npy',accontest)

    print('Saving model...')
    print(saver.save(sess, "10_model_abs/mymodel_" + str(modelnum), NumIteration))