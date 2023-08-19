import numpy as np
import os 
import tensorflow as tf
import tensorflow.contrib as contrib
from utils import  get_shape,softmax
import  sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
import math
from keras_radam.training import  RAdamOptimizer
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def rand_index(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a +=1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b +=1
            else:
                pass
    RI = (a + b) / (n*(n-1)/2)
    return RI

def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat











def BiRNN(x,n_input,n_steps,n_hidden):
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps)
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Get lstm cell output
#    try:
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                       dtype=tf.float32)
#    except Exception: # Old TensorFlow version only returns outputs not states
#        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                        dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return outputs[-1]

slim = tf.contrib.slim
eposilion = 1e-9

def batch_normal(value,is_training=False,name='batch_norm'):
    if is_training is True:
         return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)
def Active(x,mode='relu'):
    if mode == 'relu' :
        return tf.nn.relu(x)
    elif mode == 'leaky_relu' :
        return tf.nn.leaky_relu(x,alpha=0.1)
    else:
        return tf.nn.tanh(x)
mode = 'leaky_relu'

a = np.random.random((20,100,120))

def prepare_dataset(K):
	n_clusters, N, L, dt = 4, 150, 100, 0.1
	t = np.arange(0, L*dt, dt)[:L]
	seq_list, label_list = [], []
	for i in range(n_clusters):
		n_sinusoids = np.random.random_integers(1,4)
		sample_parameters = [[np.random.normal(loc=1, scale=2, size=K), np.random.normal(loc=10, scale=5, size=K)] for _ in range(n_sinusoids)]
		for j in range(N):
			seq = np.vstack([np.sum([coef[k]*np.sin(2*np.pi*freq[k]*t) for coef, freq in sample_parameters], axis=0) + np.random.randn(L) for k in range(K)]).reshape(L,K)
			seq_list.append(seq); label_list.append(i)

	return seq_list, label_list



def Attention(X,hidden):
    with tf.variable_scope("Key"):
        key = slim.fully_connected(X,hidden)
    with tf.variable_scope("query"):
        query = slim.fully_connected(X,hidden)
    with tf.variable_scope("value"):
        value = slim.fully_connected(X,hidden)
    with tf.variable_scope("Attended"):
        _,w = get_shape(query)
        query = tf.reshape(query,[w,-1])
        attend = tf.matmul(key,query)
        attend = softmax(attend,axis=-1)
    with tf.variable_scope("output"):
        outs = tf.matmul(attend,value)
        x = slim.fully_connected(X,hidden)
        x = x+ outs
    return x


def HeadCNNs(value,kernel,stride,is_train):
    head1 = tf.contrib.layers.conv1d(value,kernel,17,stride=stride,padding="SAME",activation_fn=None)
    head1 = batch_normal(head1,is_train)
    head2 = tf.contrib.layers.conv1d(value,kernel,11,stride=stride,padding="SAME",activation_fn=None)
    head2 = batch_normal(head2,is_train)
    head3 = tf.contrib.layers.conv1d(value,kernel,8,stride=stride,padding="SAME",activation_fn=None)
    head3 =  batch_normal(head3,is_train)
    head4 = tf.contrib.layers.conv1d(value,kernel,5,stride=stride,padding="SAME",activation_fn=None)
    head4 =  batch_normal(head4,is_train)
    discount = tf.contrib.layers.conv1d(value,kernel*4,9,stride=stride,padding="SAME",activation_fn=None)
    discount = batch_normal(discount,is_train)
    head = tf.concat([head1,head2,head3,head4],axis=-1)
    result = head + discount
    return Active(result,mode)

def SingleCNNs(x,kernel,channel,padding,stride,is_train):
    x = slim.conv1d(x,channel,kernel,stride=stride,padding=padding,activation_fn=None)
    x = batch_normal(x,is_training=is_train)
    x = Active(x,mode)
    return x

def HeadCNNsWithout(value,kernel,stride,is_train):
    head1 = tf.contrib.layers.conv1d(value,kernel,17,stride=stride,padding="SAME",activation_fn=None)
    head1 = batch_normal(head1,is_train)
    head2 = tf.contrib.layers.conv1d(value,kernel,11,stride=stride,padding="SAME",activation_fn=None)
    head2 = batch_normal(head2,is_train)
    head3 = tf.contrib.layers.conv1d(value,kernel,8,stride=stride,padding="SAME",activation_fn=None)
    head3 =  batch_normal(head3,is_train)
    head4 = tf.contrib.layers.conv1d(value,kernel,5,stride=stride,padding="SAME",activation_fn=None)
    head4 =  batch_normal(head4,is_train)
    discount = tf.contrib.layers.conv1d(value,kernel*4,9,stride=stride,padding="SAME",activation_fn=None)
    discount = batch_normal(discount,is_train)
    head = tf.concat([head1,head2,head3,head4],axis=-1)
    result = head + discount
    return result

def LSTM(X,unit):
  shape = get_shape(X)
  x = tf.unstack(X,shape[2],axis=2)
  lstm_cell = tf.contrib.rnn.BasicLSTMCell(unit)
  outputs , _ = tf.contrib.rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
  return outputs
################
###   LSTM Attention
###############
def LSTM_Attention(X,unit):
    with tf.variable_scope("query"):
          query =  LSTM(X,unit)[-1]
          query = tf.reshape(query,[-1,unit,1])
    with tf.variable_scope("key"):
          key = LSTM(X,unit)[-1]
          key = tf.reshape(key,[-1,1,unit])
    with tf.variable_scope("value"):
          value = LSTM(X,unit)[-1]
          value = tf.reshape(value,[-1,unit,1])
    attend = tf.matmul(query,key)
    attend = tf.nn.softmax(attend,axis=-1)
    out = tf.matmul(attend,value)
    v = slim.flatten(X)
    v = tf.reshape(slim.fully_connected(v,unit),[-1,unit,1])
    out = out + v
    return out


def Self_Attention1D(x,chanel):
    query = slim.conv1d(x,chanel,1,1)
    key = slim.conv1d(x,chanel,1,1)
    value = slim.conv1d(x,chanel,1,1)
    _ , w , _ = get_shape(x)
    key = tf.reshape(key,[-1,chanel,w])
    attend = tf.matmul(query,key)
    attend = tf.nn.softmax(attend,axis=-1)
    out = tf.matmul(attend,value)
    #print(out)
    scale = tf.constant(1.0,tf.float32)
    out = scale * out +x
    return out

def BaseDiscount(x,channel,kerl,is_train=True):
    va = slim.conv1d(x,channel,kerl,stride=1,padding="SAME",activation_fn=None)
    va = batch_normal(va,is_train)
    return va

def Encoder(X,length,num_labels,is_train=True,drop=0.5):
    with tf.variable_scope("ShapeletsNet"):
        x1 = SingleCNNs(X,11,128,"SAME",1,is_train)
        x2 = SingleCNNs(x1,11,128,"SAME",1,is_train)
        x3 = HeadCNNs(x2,32,1,is_train) 
        x4 = Self_Attention1D(x3,128)
        x5 = HeadCNNsWithout(x4,32,1,is_train)
        x6 = slim.conv1d(X,128,7,stride=1,padding="SAME",activation_fn=None)
        x6 = batch_normal(x6,is_training=is_train)
        x7 = Active(x6 + x5,mode)
        avg = tf.layers.average_pooling1d(x7,length,1,"SAME")
        avg = slim.flatten(avg)
    with tf.variable_scope("TemporalReasoning"):
        a_x =  LSTM_Attention(X,128)
        with tf.variable_scope("LSTM_attention_1"):
            l_1 = LSTM_Attention(a_x,128)
            l_1 = slim.flatten(l_1)
    output = tf.concat([l_1,avg],axis=-1)
    output_l = slim.dropout(output,drop)
    output = slim.fully_connected(output_l,num_labels,activation_fn=None)
    return output, output_l

#################### Without Attentional LSTM################
def EncoderWithoutAttentional(X,length,num_labels,is_train=True,drop=0.5):
    with tf.variable_scope("ShapeletsNet"):
        x1 = SingleCNNs(X,11,128,"SAME",1,is_train)
        x2 = SingleCNNs(x1,11,128,"SAME",1,is_train)
        x3 = HeadCNNs(x2,32,1,is_train) 
        x4 = Self_Attention1D(x3,128)
        x5 = HeadCNNsWithout(x4,32,1,is_train)
        x6 = slim.conv1d(X,128,7,stride=1,padding="SAME",activation_fn=None)
        x6 = batch_normal(x6,is_training=is_train)
        x7 = Active(x6 + x5,mode)
        avg = tf.layers.average_pooling1d(x7,length,1,"SAME")
        avg = slim.flatten(avg)
    output = avg
    output_l = slim.dropout(output,drop)
    output = slim.fully_connected(output_l,num_labels,activation_fn=None)
    return output, output_l

def WithOne(X,length,num_labels,is_train=True,drop=0.5):
    with tf.variable_scope("ShapeletsNet"):
        x1 = SingleCNNs(X,11,128,"SAME",1,is_train)
        x2 = SingleCNNs(x1,11,128,"SAME",1,is_train)
        x3 = HeadCNNs(x2,32,1,is_train) 
        x4 = Self_Attention1D(x3,128)
        x5 = HeadCNNsWithout(x4,32,1,is_train)
        x6 = slim.conv1d(X,128,7,stride=1,padding="SAME",activation_fn=None)
        x6 = batch_normal(x6,is_training=is_train)
        x7 = Active(x6 + x5,mode)
        avg = tf.layers.average_pooling1d(x7,length,1,"SAME")
        avg = slim.flatten(avg)
    with tf.variable_scope("TemporalReasoning"):
        a_x =  LSTM_Attention(X,64)
        #with tf.variable_scope("LSTM_attention_1"):
        #l_1 = LSTM_Attention(a_x,128)
        l_1 = slim.flatten(a_x)
    output = tf.concat([l_1,avg],axis=-1)
    output_l = slim.dropout(output,drop)
    output = slim.fully_connected(output_l,num_labels,activation_fn=None)
    return output, output_l

def WithThree(X,length,num_labels,is_train=True,drop=0.5):
    with tf.variable_scope("ShapeletsNet"):
        x1 = SingleCNNs(X,11,128,"SAME",1,is_train)
        x2 = SingleCNNs(x1,11,128,"SAME",1,is_train)
        x3 = HeadCNNs(x2,32,1,is_train) 
        x4 = Self_Attention1D(x3,128)
        x5 = HeadCNNsWithout(x4,32,1,is_train)
        x6 = slim.conv1d(X,128,7,stride=1,padding="SAME",activation_fn=None)
        x6 = batch_normal(x6,is_training=is_train)
        x7 = Active(x6 + x5,mode)
        avg = tf.layers.average_pooling1d(x7,length,1,"SAME")
        avg = slim.flatten(avg)
    with tf.variable_scope("TemporalReasoning"):
        a_x =  LSTM_Attention(X,64)
        with tf.variable_scope("LSTM_attention_1"):
            l_1 = LSTM_Attention(a_x,128)
        with tf.variable_scope("LSTM_attention_2"):
            l_1 = LSTM_Attention(l_1,128)
            l_1 = slim.flatten(l_1)
    output = tf.concat([l_1,avg],axis=-1)
    output_l = slim.dropout(output,drop)
    output = slim.fully_connected(output_l,num_labels,activation_fn=None)
    return output, output_l

def super_soft_encross(predic,label):
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(label,predic))
#
#
def Optimazer(loss,choose=1,lr=0.001,decay=200):
    global_step =  tf.Variable(0, name='global_step')
    lr = tf.train.exponential_decay(lr, global_step, decay, 0.9, staircase=True)
    if choose == 1:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step)
    elif choose == 2 :
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss,global_step)
    elif choose == 3 :
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step)
    else :
        train_op = RAdamOptimizer(lr).minimize(loss,global_step)
    return train_op


def DerectOptimzer(loss,choose=1,lr=0.0001):
    global_step =  tf.Variable(0, name='global_step')
    if choose == 1:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step)
    elif choose == 2 :
        train_op = tf.train.RMSPropOptimizer(lr,decay=0.9,momentum=0.0).minimize(loss,global_step)
    elif choose == 3 :
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step)
    else :
        train_op = RAdamOptimizer(lr).minimize(loss,global_step)
    return train_op


def SquareLoss(predic,label):
    square_loss = tf.reduce_mean(tf.square(predic-label))
    return square_loss


def Decoder(encode,length):
    x1 = slim.fully_connected(encode,256,activation_fn=tf.nn.leaky_relu)
    x2 = slim.fully_connected(x1,512,activation_fn=tf.nn.leaky_relu)
    x3 = slim.fully_connected(x2,512,activation_fn=tf.nn.leaky_relu)
    x4 = slim.fully_connected(x3,length,activation_fn=tf.sigmoid)
    return x4


def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost



