import processing.config as cfg
import data.data as data
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sklearn
import matplotlib.pyplot as plt
from processing.utils import reduce_sum
from processing.utils import softmax
from processing.utils import get_shape,Totalcount
import  processing.config as cfg
import  time
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
import model.networks as net
import sklearn.metrics as mertrics
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
arser = argparse.ArgumentParser()
parser.add_argument("-n","--name",type=int,help='training file names'  )
parser.add_argument("-b","--batch",type=int,help="integrates for training and testing",default=1)
parser.add_argument("-r","--ratio",type=float,help="ratio range from 0.1 to 0.3",default=0.1)
parser.add_argument("-l","--init_learning",type=float,help="initial learning rate",default=0.0001)
parser.add_argument("-s","--decay_steps",type=int,help="decay steps",default=300)
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

length = x_train.shape[1]

#######################################################
####
train_batch = x_train.shape[0]//b

test_batch = x_test.shape[0]//b
#    
#############################
X = tf.placeholder(tf.float32,[None,length,dims])
Y = tf.placeholder(tf.float32,[None,num_classes])
is_train = tf.placeholder(tf.bool)
drop = tf.placeholder(tf.float32)
###########################################
###########################
#logits,features = net.BuildResNetLSTMaN(X,num_classes,length,drop,is_train)
logits,stores = net.DKN(X,32,length,num_classes,1,drop,is_train) 
##################
eachlists = [net.AdaptiveClassifier(v,num_classes,drop) for v in stores]
lossself = 0
T = 1.0

for result in eachlists:
    lossself += net.soft_kl_divergence_with_logits( tf.stop_gradient(logits)/T, result/T ) + net.soft_kl_divergence_with_logits( tf.stop_gradient(result)/T, logits/T )

#print(len(eachlists))

for i in range(5,7):
    lossself += net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[7])/T, eachlists[i]/T ) + net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[i])/T, eachlists[7]/T )

lossself += net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[6])/T, eachlists[5]/T ) + net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[5])/T, eachlists[6]/T )

for i in range(0,4):
    lossself += net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[4])/T, eachlists[i]/T ) + net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[i])/T, eachlists[4]/T )

for i in range(0,3):
    lossself += net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[3])/T, eachlists[i]/T ) + net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[i])/T, eachlists[3]/T )

for i in range(0,2):
    lossself += net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[2])/T, eachlists[i]/T ) + net.soft_kl_divergence_with_logits( tf.stop_gradient(eachlists[i])/T, eachlists[2]/T )


#exit(0)

wd_weight = 0.0005
loss_wd = tf.add_n( [tf.nn.l2_loss(tf.cast(v,tf.float32)) for v in tf.trainable_variables()  ]   )
#loss_wd = sum(tf.nn.l2_loss(v) for v in net.model_vars('classifier') if 'kernel' in v.name)
base_loss = 0

for result in eachlists:
    base_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(Y,result))


loss = (1-r)*(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(Y,logits))+ base_loss) + r* lossself + 0.0005*loss_wd

train_op,lr = net.Optimazer(loss,2,init_learn,steps=decay_steps)

############################ Evaluation###################
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(logits,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
########################################################


#exit(0)
################ 184 618 618



train_epoch = x_train.shape[0] // train_batch
test_epoch = x_test.shape[0] // test_batch
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(tf.global_variables())
#saver.restore(sess,"data/"+class_name+".ckpt")
TotalPara = net.Totalcount()
start_time = time.time()
total_loss_npd = []
for i in range(epoch):
    for j in range(train_epoch):
        if j != train_epoch-1:
                    batch_x = x_train[train_batch*j:(j+1)*train_batch,...]
                    batch_y = y_train[train_batch*j:(j+1)*train_batch,...]
                    _, total_loss = sess.run(
                    [train_op,loss],feed_dict={X:batch_x,Y:batch_y,is_train:True,drop:0.5}
                     )
        else:
                    batch_x = x_train[train_batch*j:]
                    batch_y = y_train[train_batch*j:]
                    _, total_loss = sess.run(
                        [train_op,loss],feed_dict={X:batch_x,Y:batch_y,is_train:True,drop:0.5}
                             )
    total_loss_npd.append(total_loss)
    print(i+1,'-----------------------------------------------------------------training loss:%.6f----------'%(total_loss))
print("Spends:%.4fs"%(time.time()-start_time))
sess.close()