from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import cv2
from keras.datasets import cifar10
from scipy import misc
import os
import copy


for modelnum in range(0,10):
    for attacknum in range(10):
        tf.reset_default_graph()
        # Parameters
        BatchLength = 20
        NumIteration = 10001;
        LearningRate = 1e-4# learning rate of the algorithm
        NumClasses = 2 # number of output classes
        EvalFreq = 100  # evaluate on every 100th iteration
        NumOfChannels=[3,64,128] #number of channels/layer including the input picture (3:RGB,4:RGBA)
        Size=[32,32,3]
        ImSize=(Size[1],Size[0])
        AbSize=[32,32,1]
        Stride=1
        Pooling=2
        imgMode='RGB'
        Abmode='F'
        ar_len=10000
        factor=4
        iteration='60001'

        (TrainImages, TrainLabels), (TestImages, TestLabels) = tf.keras.datasets.cifar10.load_data()

        gfile='mymodel_'
        gpath='./10_model_abs/'
        metafile=gfile+str(modelnum)+'-'+iteration+'.meta'
        indexfile=gfile+str(modelnum)+'-'+iteration



        Label_0_im=np.zeros([ar_len,Size[0],Size[1],Size[2]])
        Label_1_im=np.zeros([ar_len,Size[0],Size[1],Size[2]])

        Label_0_lab=np.zeros([ar_len])
        Label_1_lab=np.ones([ar_len])


        counter=0
        lab0=0
        lab1=0
        for b,label in enumerate(TestLabels):
            if label==0.0:
                Label_0_lab[lab0]=0
                image=TestImages[b]
                Label_0_im[lab0]=image
                counter+=1
                lab0+=1
            if label==7.0:
                Label_1_lab[lab1]=1
                image=TestImages[b]
                Label_1_im[lab1]=image
                counter+=1
                lab1+=1
        Label_0_im=Label_0_im[0:lab0,:,:,:]
        Label_0_lab=Label_0_lab[0:lab0]

        Label_1_im=Label_1_im[0:lab1,:,:,:]
        Label_1_lab=Label_1_lab[0:lab1]


        saver = tf.train.import_meta_graph(gpath+metafile)


        #Loading Tensors
        graph = tf.get_default_graph()
        InputData = graph.get_tensor_by_name("TrainInput:0")
        outputImage = graph.get_tensor_by_name("Output:0")
        predictedlabel=graph.get_tensor_by_name("accuracy/PredictedLabel:0")
        accloss=graph.get_tensor_by_name('accuracy/AccLoss:0')
        InputLabels=graph.get_tensor_by_name("DesiredLabelOutput:0")
        accuracy=graph.get_tensor_by_name("accuracy/Accuracy:0")

        #############################################################################################ENDOFFRANCISKA

        # Initializing the variables
        Init = tf.global_variables_initializer()
        locinit=tf.local_variables_initializer()


        howmanybatches = len(Label_1_im) / BatchLength
        howmanybatches = int(howmanybatches)
        print(howmanybatches)
        successcounter=np.zeros([howmanybatches,BatchLength])


        # Launch the session with default graph
        with tf.Session() as Sess:
            Sess.run(Init)
            saver.restore(Sess, gpath+indexfile)

            # test
            for batch_step in range(0, howmanybatches * BatchLength, BatchLength):
                print('\n Ez a ' + str(batch_step // BatchLength) + '. batch kezdete. (' + str(howmanybatches) + '-ből.)')
                OrigData_batch = Label_1_im[batch_step:batch_step + BatchLength, :, :, :]
                Label_batch = Label_1_lab[batch_step:batch_step + BatchLength]
                for inbatch in range(0, BatchLength):
                    OrigData = np.expand_dims(OrigData_batch[inbatch], 0)
                    Label = np.expand_dims(Label_batch[inbatch], 0)

                    Pred = Sess.run(accloss, feed_dict={InputData: OrigData, InputLabels: Label})
                    Pred = Pred[0]
                    print('kezdeti pred: ' + str(Pred))
                    GoalClass = 0
                    Label_ = int(Label)

                    if Pred[Label_] < Pred[GoalClass]:

                        MinWeight = Pred[Label_] - Pred[GoalClass]
                        MinPos = []
                        MinSize = []
                        InitialValue = (MinWeight)
                        # we have two white and two black stickers
                        GenomeSize = 100
                        NumSteps = 20
                        KeepRatio = 0.2
                        NewRatio = 0.1
                        StickerNum = 5
                        StickerColor = [0.0, 255.0, 0.0, 255.0,0.0]
                        MaxStickerSize = 5
                        MinStickersize = 0
                        GeneratedRatio = 1 - (KeepRatio + NewRatio)
                        MutationFactor = 0.2

                        Positions = np.zeros((GenomeSize, StickerNum * 2))
                        Sizes = np.zeros((GenomeSize, StickerNum * 2))

                        # generate Initial Genome
                        for i in range(GenomeSize):
                            Positions[i, :] = np.random.uniform(0, Size[1], StickerNum * 2)
                            Sizes[i, :] = np.random.uniform(MinStickersize, MaxStickerSize, StickerNum * 2)
                        for St in range(NumSteps):
                            # calc weights
                            Weights = np.zeros(GenomeSize)

                            StickerData_ow = np.copy(OrigData)
                            StickerData_ow = StickerData_ow[0].astype('float32')
                            StickerData_ow = cv2.cvtColor(StickerData_ow, cv2.COLOR_RGB2BGR)

                            for genom_step in range(0, GenomeSize, BatchLength):
                                StickerData_batch = np.zeros(shape=(BatchLength, *OrigData[0].shape))
                                for ingenom in range(BatchLength):
                                    StickerData_batch[ingenom, :, :, :] = np.copy(OrigData)

                                for ingenom in range(BatchLength):
                                    for s in range(StickerNum):
                                        i = genom_step + ingenom
                                        StickerData_batch[ingenom,
                                        int(Positions[i, 2 * s]):int(Positions[i, 2 * s] + Sizes[i, 2 * s]),
                                        int(Positions[i, (2 * s) + 1]):int(Positions[i, (2 * s) + 1] + Sizes[i, (2 * s) + 1]), :] = \
                                            StickerColor[s]

                                Pred_batch = Sess.run(accloss, feed_dict={InputData: StickerData_batch})

                                for ingenom in range(BatchLength):
                                    i = genom_step + ingenom
                                    Pred = Pred_batch[ingenom]
                                    StickerData = StickerData_batch[ingenom]
                                    Weights[i] = Pred[Label_] - Pred[GoalClass]
                                    if Weights[i] > MinWeight:
                                        MinWeight = Weights[i]
                                        MinPos = Positions[i, :]
                                        MinSize = Sizes[i, :]
                                        StickerData_ = StickerData.astype('float32')
                                        StickerData_ = cv2.cvtColor(StickerData_, cv2.COLOR_RGB2BGR)
                                        if Pred[Label_] > Pred[GoalClass]:
                                            print('!ÁTVERÉS >:( !: ' + str(Pred))
                                            cv2.imwrite(
                                                './new_ab/' + str(batch_step).zfill(3) + '_' + str(inbatch).zfill(3) + 'orig.png',
                                                StickerData_ow)
                                            cv2.imwrite('./new_ab/' + str(batch_step).zfill(3) + '_' + str(inbatch).zfill(
                                                3) + 'success_' + str(StickerNum) + '_sticker.png', StickerData_)
                                            successcounter[batch_step // BatchLength][inbatch] += 1
                                            nonzero_ = np.count_nonzero(successcounter)
                                            currentstep_ = batch_step + inbatch + 1
                                            atlag_ = nonzero_ / currentstep_
                                            print('Az eddigi állás szerint a háló ' + str(
                                                atlag_ * 100) + ' százalékban átverődött. :( Átverések száma: '+str(nonzero_))
                                            break;
                                if Pred[Label_] > Pred[GoalClass]:
                                    break;
                            if Pred[Label_] > Pred[GoalClass]:
                                break;
                            # order the Population
                            Indices = range(GenomeSize)
                            Weights, Indices = zip(*sorted(zip(Weights, Indices)))
                            KeptIndices = Indices[0:int(KeepRatio * GenomeSize)]
                            GeneratedIndices = int((1.0 - NewRatio) * GenomeSize)
                            NewPositions = np.zeros((GenomeSize, 2 * StickerNum))
                            NewSizes = np.zeros((GenomeSize, 2 * StickerNum))
                            # elitism - keep the best elements
                            for a in range(len(KeptIndices)):
                                NewPositions[a, :] = Positions[KeptIndices[a], :]
                                NewSizes[a, :] = Sizes[KeptIndices[a], :]
                            # crossover for the generated ones
                            for a in range(len(KeptIndices), GeneratedIndices):
                                # select two samples
                                Indices = np.random.choice(range(len(KeptIndices)), 2, replace=False)
                                # select point of the crossover
                                CrossPoint = np.random.randint(0, (2 * StickerNum) + 1)
                                NewPositions[a, 0:CrossPoint] = Positions[KeptIndices[Indices[0]]][0:CrossPoint]
                                NewPositions[a, CrossPoint:2 * StickerNum] = Positions[KeptIndices[Indices[1]]][
                                                                             CrossPoint:2 * StickerNum]
                                NewSizes[a, 0:CrossPoint] = Sizes[KeptIndices[Indices[0]]][0:CrossPoint]
                                NewSizes[a, CrossPoint:2 * StickerNum] = Sizes[KeptIndices[Indices[1]]][CrossPoint:2 * StickerNum]
                            # rest is new
                            for a in range(len(KeptIndices), GenomeSize):
                                NewPositions[a, :] = np.random.uniform(0, Size[1], 2 * StickerNum)
                                NewSizes[a, :] = np.random.uniform(MinStickersize, MaxStickerSize, 2 * StickerNum)

                            # random mutation
                            for a in range(GenomeSize):
                                ranun = np.random.uniform()
                                if ranun < MutationFactor:
                                    NewPositions[a, :] += np.random.normal(0, 3, 2 * StickerNum)
                                    NewSizes[a, :] += np.random.normal(0, 3, 2 * StickerNum)
                                    for i in range(2 * StickerNum):
                                        if NewSizes[a, i] > MaxStickerSize:
                                            NewSizes[a, i] = MaxStickerSize
                                        if NewSizes[a, i] < MinStickersize:
                                            NewSizes[a, i] = MinStickersize

                            Positions = NewPositions
                            Sizes = NewSizes
                        print("Initial Value: " + str(InitialValue) + " Optimized Value: " + str(MinWeight))
                    else:
                        successcounter[batch_step // BatchLength][inbatch] += -1

            print(MinWeight)
            print(MinPos)
            print(MinSize)


        np.save('./5stck_abs_model/successcounter_model' +str(modelnum)+'_attack'+str(attacknum)+'_konv.npy',successcounter)