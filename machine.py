import config as cfg
import data 
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sklearn
import matplotlib.pyplot as plt
from utils import reduce_sum
from utils import softmax
from utils import get_shape,Totalcount
import  config as cfg
import  time
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
import networks as net
import sklearn.metrics as mertrics
import argparse
import object as obj
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as mertrics
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
arser = argparse.ArgumentParser()
parser.add_argument("-n","--name",type=int,help='training file names'  )
parser.add_argument("-b","--batch",type=int,help="integrates for training and testing",default=1)
parser.add_argument("-r","--ratio",type=float,help="ratio range from 0.1 to 0.3",default=0.1)
parser.add_argument("-l","--init_learning",type=float,help="initial learning rate",default=0.0001)
parser.add_argument("-s","--decay_steps",type=int,help="decay steps",default=200)
parser.add_argument("-p","--option",type=int,help="option for selecting the optimizer",default=1)
parser.add_argument("-c","--confidence",type=float,help="confidence for TiMatch",default=0.95)
parser.add_argument("-e","--epochs",type=int,help="the number of training epochs",default=2000)
parser.add_argument("-k","--expand",type=float,help="the number of augmentations",default=2)
arg = parser.parse_args()



infor = data.LoadMTSCUEA(arg.name)
#23
#28 None
#8 9  21 23 26
x_train = infor["TrainX"]
y_train = infor["TrainY"]
name = infor["Name"]
x_test = infor["TestX"]
y_test = infor["TestY"]


b = arg.batch
r = arg.ratio
training_name = name
init_learn = arg.init_learning
decay_steps = arg.decay_steps
 ##### TrainSize
dims = x_train.shape[-1]
k=arg.expand
ratio = str(r*100)
class_name = training_name
confidence = arg.confidence

epoch = arg.epochs



#total_data = np.concatenate([x_train,x_test],axis=0)
#total_label = np.concatenate([y_train,y_test],axis=0)
#x_train,x_test,y_train,y_test = train_test_split(total_data,total_label,test_size=0.8,random_state=total_data.shape[0])
num_classes = y_test.shape[1]
#y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(num_classes-1)

x_test = (x_test - x_test.mean(axis=0))/(x_test.std(axis=0))
x_train = (x_train-x_train.mean(axis=0))/(x_train.std(axis=0))
trainbatch = x_train.shape[0]
testbatch = x_test.shape[0]
x_train = x_train.reshape((trainbatch,x_test.shape[1] * x_test.shape[2]   ))
x_test = x_test.reshape((testbatch,  x_test.shape[1]*x_test.shape[2]  ))

KNN = KNeighborsClassifier()
SVM = SVC()
RM = RandomForestClassifier()
GB = GradientBoostingClassifier()
LR  = LogisticRegression()
DT = DecisionTreeClassifier()

y_train = np.argmax(y_train,axis=-1)
y_test = np.argmax(y_test,axis=-1)
KNN.fit(x_train,y_train)
SVM.fit(x_train,y_train)
RM.fit(x_train,y_train)
GB.fit(x_train,y_train)
LR.fit(x_train,y_train)
DT.fit(x_train,y_train)

print(class_name)

print("KNN:f1-score:",mertrics.accuracy_score(y_test,KNN.predict(x_test)))
print("SVM:f1-score:",mertrics.accuracy_score(y_test,SVM.predict(x_test)))
print("RM:f1-score:",mertrics.accuracy_score(y_test,RM.predict(x_test)))
print("GB:f1-score:",mertrics.accuracy_score(y_test,GB.predict(x_test)))
print("LR:f1-score:",mertrics.accuracy_score(y_test,LR.predict(x_test)))
print("DT:f1-score:",mertrics.accuracy_score(y_test,DT.predict(x_test)))

