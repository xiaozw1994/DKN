import numpy as np
import os 
import matplotlib.pyplot as plt
from keras.utils import  np_utils
import tensorflow as tf
import random
#
#
# reading data from the txt files
#
save_file =  "/mnt/54b93d63-868e-4b2c-aa09-c9109c3e67be/multivariateTSC/UCE"
total=['SpokenArabicDigits', 'UWaveGestureLibrary', 'BasicMotions',
 'NATOPS', 'DuckDuckGeese', 'FingerMovements', 'PEMS-SF', 
 'PhonemeSpectra', 'CharacterTrajectories', 'Libras', 'Handwriting', 
 'StandWalkJump', 'ERing', 'Cricket', 'Epilepsy', 
 'ArticularyWordRecognition', 'AtrialFibrillation', 'SelfRegulationSCP2', 
 'SelfRegulationSCP1', 
'RacketSports', 'MotorImagery', 'LSST', 'InsectWingbeat',
 'JapaneseVowels', 'Heartbeat', 'PenDigits', 'FaceDetection', 'EigenWorms', 
 'HandMovementDirection', 'EthanolConcentration']
train_nameX_suffix = "_trainx"
train_nameY_suffix = "_trainy"
test_nameX_suffix = "_testx"
test_nameY_suffix = "_testy"

#################################
####   EEG 6 7 18 19 21 26 27 29
################################
suffix = ".npy"

EEG_list = [6,7,18,19,21,26,27,29]


def LoadMTSCUEA(index):
    ##### 1 --- 36
    if index > len(total):
        raise EOFError("Eorror index")
    else:
        infor = {
            "TrainX":None,
            "TrainY":None,
            "TestX":None,
            "TestY":None,
            "Name":None,
        }
        name = total[index-1]
        infor["Name"] = name
        trainx = np.load( os.path.join(save_file,name+train_nameX_suffix+suffix))
        trainy = np.load(os.path.join(save_file,name+train_nameY_suffix+suffix))
        testx = np.load(os.path.join(save_file,name+test_nameX_suffix+suffix))
        testy = np.load(os.path.join(save_file,name+test_nameY_suffix+suffix))
        infor["TrainX"] = trainx
        infor["TrainY"] = trainy
        infor["TestX"] = testx
        infor["TestY"] = testy
        return infor


def show(name, data):
    #data = (X,dimension)
    dimension = data.shape[1]
    x = [i for i in range(1,data.shape[0]+1)]
    plt.title(name)
    for i in range(dimension):
        plt.plot(x,data[:,i],label="Dimension_"+str(i+1))
    plt.legend()
    plt.show()

def jitter(x,sigma=0.8):
    return x + np.random.normal(loc=0,scale=sigma,size=x.shape)

def scaling(x,sigma=1.1):
    factor = np.random.normal(loc=2,scale=sigma,size=(x.shape[0],x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:,i,:]
        ai.append(  np.multiply(xi, factor[:,:])[:,np.newaxis,:]        )
    return np.concatenate((ai),axis=1)

def generalAugmentation(x):
    weak_aug = scaling(x)
    strong_aug = jitter(x)
    return weak_aug,strong_aug









def readucr(filename):
    data = np.loadtxt(filename+".tsv", delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    if np.isnan(X).any():
        X[np.isnan(X)] = np.nanmean(X)
    return X, Y
###
#    
#  To normalize the trained lebeled data
def NormalizationClassification(Y,num_classes):
    Y = np.array(Y)
    return (Y-Y.mean()) / (Y.max()-Y.mean()) *(num_classes-1)
#
#
#
def NormalizationFeatures(X):
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    value = (X-mean) / std
    return value
#
#  This file is used to get the size of class in the training dataset
#
def GetNumClasses(y):
    y = np.array(y)
    num_classes = len(np.unique(y))
    return num_classes
#### Noising
#   Using the Gussian function 
#

#
#   To OneHot
#
def OneHot(y,num_classes):
    y = np.array(y)
    y = np_utils.to_categorical(y,num_classes)
    return y
#
#
#Show The index of picure
#
def Show(train_x,aug_x,index,length):
    x = [i for i in range(1,length+1)]
    fig = plt.figure()
    aix = fig.subplots(nrows=2,ncols=1)
    aix[0].plot(x,train_x[index])
    aix[1].plot(x,aug_x[index])
    plt.show()
#
#
# Agumentation 
#

def showRand(train_x,length):
    index = np.random.randint(0,length)
    l = 6
    x = [i for i in range(1,train_x.shape[1]+1)]
    fig = plt.figure()
    aix = fig.subplots(nrows=l,ncols=1)
    for i in range(0,l):
        aix[i].plot(x,train_x[i*length+index,...])
    plt.show()

############################
########## Load Files
############################

#########################################
#  Time Series Cutout            

def TimeCutout(feature,pad_h,replace=0):
    '''
    args:
        feature : the raw data. [length,1]
        pad_h : padding size
    '''
    length = tf.shape(feature)[0]
    coutout_center_length  = tf.random_uniform(shape=[],minval=0,maxval=length,dtype=tf.int32)
    lower_pad = tf.maximum(0,coutout_center_length-pad_h)
    uperr_pad = tf.maximum(0,length-coutout_center_length-pad_h)
    cutout_shape = [
        length-(lower_pad+uperr_pad),1
    ]
    padding_dims = [[lower_pad,uperr_pad],[0,0]]
    mask = tf.pad(
        tf.zeros(cutout_shape,dtype=feature.dtype),
        padding_dims,constant_values=1
    )
    feature = tf.where(
        tf.equal(mask,0),tf.ones_like(feature,dtype=feature.dtype)*replace,feature
    )
    return feature
##############################
######################################################
def TimeCutoutNumpy(feature,pad,replace=0):
    '''
    args:
        feature : the raw data. [length,1]
        pad : padding size
    '''
    length = feature.shape[0]
    coutout_center_length = np.random.randint(low=0,high=length)
    lower_padd = np.maximum(0,coutout_center_length-pad)
    upper_padd = np.maximum(0,length-coutout_center_length-pad)
    cutout_shape = [
        length-(lower_padd+upper_padd),1
    ]
    pad_dims = [[lower_padd,upper_padd],[0,0]]
    mask = np.pad(
        np.zeros(cutout_shape,dtype=feature.dtype),
        pad_dims,'constant',constant_values=1
    )
    feature = np.where(
        np.equal(mask,0),np.ones_like(feature,dtype=feature.dtype)*replace,feature
    )
    return feature

def TimeCutoutSingle(feature,pad,cutout,replace=0):
    '''
    args:
        feature : the raw data. [length,1]
        pad : padding size
    '''
    length = feature.shape[0]
    coutout_center_length = cutout
    lower_padd = np.maximum(0,coutout_center_length-pad)
    upper_padd = np.maximum(0,length-coutout_center_length-pad)
    cutout_shape = [
        length-(lower_padd+upper_padd),1
    ]
    pad_dims = [[lower_padd,upper_padd],[0,0]]
    mask = np.pad(
        np.zeros(cutout_shape,dtype=feature.dtype),
        pad_dims,'constant',constant_values=1
    )
    feature = np.where(
        np.equal(mask,0),np.ones_like(feature,dtype=feature.dtype)*replace,feature
    )
    return feature

####################### Policy################
def Policy(Batch,length):
    ###
    ##
    '''
    Batch : represents the number of sample
    length: the length of  time series dataset
    '''
    highlength = Batch//2
    lowlength = Batch - highlength
    highrate = np.random.uniform(low=0.5,high=1.0,size=1)
    lowrate = np.random.uniform(low=0.0,high=0.5,size=1)
    decay = 0.99
    rates = []
    for i in range(highlength):
        highrate = highrate * decay + (1-highrate) * (1-decay)
        if highrate < 0.4:
            highrate =  np.random.uniform(low=0.5,high=1.0,size=1)
        else :
            rates.append(highrate[0])
    for i in range(lowlength):
        lowrate = lowrate * decay  
        if lowrate < 0.0:
            lowrate = np.random.uniform(low=0.0,high=0.5,size=1)
        else :
            rates.append(lowrate[0])
    rates = [int(i*length) for i in rates]
    return rates
def AutoAugmentation(unlabeled):
    ###############
    ####
    batch = unlabeled.shape[0]
    length = unlabeled.shape[1]
    unlabeled = unlabeled.reshape((batch,length,1))
    rates = Policy(batch,length)
    feat = []
    for i in range(len(rates)):
        rate = rates[i]
        unlabel = unlabeled[i]
        pad = np.random.randint(0,5,1)[0]
        feature = TimeCutoutSingle(unlabel,pad,rate)
        feat.append(feature)
    feat = np.array(feat).reshape((-1,length))
    return feat
######################Augmentation Operations##########################################
def SemiAugmentation(unlabeled,k=1):
    if k==1:
        raw_data = unlabeled
        auto = AutoAugmentation(unlabeled)
    else :
        raw_data = []
        auto = []
        raw_data.append(unlabeled)
        for i in range(k):
            au = AutoAugmentation(unlabeled)
            auto.append(au)  
            if i >0  :
                para = np.random.uniform(low=0.05,high=0.2,size=2)
                feat =  np.random.normal(loc=para[0],scale=para[1],size=unlabeled.shape)+unlabeled
                raw_data.append(feat)
        raw_data,auto = np.concatenate(raw_data,0),np.concatenate(auto,0)
    return raw_data,auto
