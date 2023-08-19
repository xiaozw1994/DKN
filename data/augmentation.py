import numpy as np
import random


#################################
########  add Guassian Functions
#################################
def addNoisy(x, loc,mean):
    return x+ np.random.normal(loc=loc,scale=mean,size=x.shape)
#################################
###### Timeout 
##################################################################
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
    patt = feature.shape[1]
    lower_padd = np.maximum(0,coutout_center_length-pad)
    upper_padd = np.maximum(0,length-coutout_center_length-pad)
    cutout_shape = [
        length-(lower_padd+upper_padd),patt
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
def TimeoutSequential(unlabeled):
    ###############
    ####
    batch = unlabeled.shape[0]
    length = unlabeled.shape[1]
    patt = unlabeled.shape[2]
    unlabeled = unlabeled.reshape((batch,length,patt))
    rates = Policy(batch,length)
    feat = []
    for i in range(len(rates)):
        rate = rates[i]
        unlabel = unlabeled[i]
        pad = np.random.randint(0,5,1)[0]
        feature = TimeCutoutSingle(unlabel,pad,rate)
        feat.append(feature)
    feat = np.array(feat).reshape((-1,length,patt))
    return feat

def scaling(x,sigma=1.1):
    factor = np.random.normal(loc=2,scale=sigma,size=(x.shape[0],x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:,i,:]
        ai.append(  np.multiply(xi, factor[:,:])[:,np.newaxis,:]        )
    return np.concatenate((ai),axis=1)

def RandomPartion(unlabeled,ratio=0.9,replace=np.nan):
    batch = unlabeled.shape[0]
    length = unlabeled.shape[1]
    patt = unlabeled.shape[2]
    start = np.random.randint(low=0,high=  int( (1-ratio)*length  ),size=1  )[0]
    end = int(start+ int(length*ratio))
    feature = []
    for i in range(batch):
        each_data = np.random.random(size=(1,length,patt))
        each_data[:,start:end,...] = unlabeled[i,start:end,...]
        if start != 0:
            each_data[:,0:start,...] = replace
        each_data[:,end:,...] = replace
        feature.append(each_data)
    feat = np.array(feature).reshape((batch,length,patt))
    return feat

def Check(X):
     if np.isnan(X).any():
        X[np.isnan(X)] = np.nanmean(X)
     return X



