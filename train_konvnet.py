from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
from keras.datasets import cifar10

for modelnum in range(0,40):

    tf.reset_default_graph()
    # Parameters
    BatchLength = 20  # 32 images are in a minibatch
    Size = [32, 32, 3]  # Input img will be resized to this size
    NumIteration = 10001;
    LearningRate = 1e-4  # learning rate of the algorithm
    NumClasses = 2
    NumKernels = [64, 128, 64]
    fckern=128

    # load cifar data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    k = 0
    for b, label in enumerate(y_train):
        if label == 0.0:
            y_train[k] = 0
            image = x_train[b]
            x_train[k] = image
            k += 1
        if label == 7.0:
            y_train[k] = 1
            image = x_train[b]
            x_train[k] = image
            k += 1
    x_train = x_train[0:k, :, :, :]
    y_train = y_train[0:k]

    j = 0
    for b, label in enumerate(y_test):
        if label == 0.0:
            y_test[j] = 0
            image = x_test[b]
            x_test[j] = image
            j += 1
        if label == 7.0:
            y_test[j] = 1
            image = x_test[b]
            x_test[j] = image
            j += 1
    x_test = x_test[0:j, :, :, :]
    y_test = y_test[0:j]

    # Create tensorflow graph
    InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]],name='TrainInput')  # network input
    InputLabels = tf.placeholder(tf.int32, [None],name='DesiredLabelOutput')  # desired network output
    OneHotLabels = tf.one_hot(InputLabels, NumClasses)




    def MakeConvNet(Input, Size):
        CurrentInput = Input
        CurrentFilters = Size[2]  # the input dim at the first layer is 1, since the input image is grayscale
        for i in range(len(NumKernels)):  # number of layers
            with tf.variable_scope('conv' + str(i)):
                NumKernel = NumKernels[i]
                W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel])

                CurrentFilters = NumKernel
                ConvResult = tf.nn.conv2d(CurrentInput, W, strides=[1, 1, 1, 1], padding='VALID')  # VALID, SAME
                #add batch normalization
                beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))
                gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))
                Mean,Variance = tf.nn.moments(ConvResult,[0,1,2])
                PostNormalized = tf.nn.batch_normalization(ConvResult,Mean,Variance,beta,gamma,1e-10)

                ReLU = tf.nn.relu(PostNormalized)


                CurrentInput = tf.nn.max_pool(ReLU, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # add fully connected network
        with tf.variable_scope('FC'):
            CurrentShape = CurrentInput.get_shape()
            FeatureLength = int(CurrentShape[1] * CurrentShape[2] * CurrentShape[3])
            FC = tf.reshape(CurrentInput, [-1, FeatureLength])
            W = tf.get_variable('W', [FeatureLength, fckern])
            FC = tf.matmul(FC, W)
            Bias = tf.get_variable('Bias', [fckern])
            CurrentInput = tf.add(FC, Bias)
            CurrentInput = tf.nn.relu(CurrentInput)
        with tf.variable_scope('Logit'):
            CurrentShape = CurrentInput.get_shape()
            FC = tf.reshape(CurrentInput, [-1, fckern])
            W = tf.get_variable('W', [fckern, NumClasses])
            FC = tf.matmul(FC, W)
            Bias = tf.get_variable('Bias', [NumClasses])
            FC = tf.add(FC, Bias,name='Output')
        return FC


    # Construct model
    PredWeights = MakeConvNet(InputData, Size)


    # Define loss and optimizer
    with tf.name_scope('loss'):
        Loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(OneHotLabels, PredWeights))

    with tf.name_scope('optimizer'):
        # Use ADAM optimizer this is currently the best performing training algorithm in most cases
        Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
        # Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

    with tf.name_scope('accuracy'):
        CorrectPredictions = tf.equal(tf.argmax(PredWeights, 1), tf.argmax(OneHotLabels, 1))
        Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32),name='Accuracy')

    # Initializing the variables
    Init = tf.global_variables_initializer()

    # Launch the session with default graph
    with tf.Session() as Sess:
        Sess.run(Init)
        Saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        Step = 1
        # Keep training until reach max iterations - other stopping criterion could be added
        while Step < NumIteration:

            # create train batch - select random elements for training
            TrainIndices = random.sample(range(x_train.shape[0]), BatchLength)
            Data = x_train[TrainIndices, :, :, :]
            Label = y_train[TrainIndices]
            Label = np.reshape(Label, (BatchLength))

            # execute the session
            _, Acc, L = Sess.run([Optimizer, Accuracy, Loss], feed_dict={InputData: Data, InputLabels: Label})

            # print loss and accuracy at every 10th iteration
            if (Step % 10) == 0:
                # train accuracy
                print("Iteration: " + str(Step))
                print("Accuracy:" + str(Acc))
                print("Loss:" + str(L))
            Step += 1

        # independent test accuracy
        TotalAcc = 0.0
        Iter = 0
        Data = np.zeros([BatchLength] + Size)
        for i in range(0, x_test.shape[0] - BatchLength, BatchLength):
            Data = x_test[i:(i + BatchLength), :, :, :]
            Label = y_test[i:(i + BatchLength)]
            Label = np.reshape(Label, (BatchLength))
            response, acc = Sess.run([PredWeights, Accuracy], feed_dict={InputData: Data, InputLabels: Label})
            TotalAcc += acc
            Iter += 1
            accontest=float(TotalAcc) / float(Iter)
        print("Independent Test set: " + str(float(TotalAcc) / float(Iter)))
        np.save('./model_acc_bn/accuracy_'+str(modelnum)+'.npy',accontest)

        print('Saving model...')
        print(Saver.save(Sess, "10_model_bn/mymodel_"+str(modelnum), Step))

    print("Optimization Finished!")


