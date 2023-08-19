import tensorflow as tf
import  numpy as np
from processing.utils import reduce_sum
from processing.utils import softmax
from processing.utils import get_shape

slim = tf.contrib.slim


eposilion = 1e-9

def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def batch_normal(value,is_training=False,name='batch_norm'):
    if is_training is True:
         return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)
def Active(x,mode='relu'):
    if mode == 'relu' :
        return tf.nn.relu(x)
    elif mode == 'leaky_relu' :
        return tf.nn.leaky_relu(x,alpha=0.1)
    else:
        return tf.nn.tanh(x)
mode = 'relu'

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


def baseResidual(X,num_features,is_train):
    mode = 'relu'
    x1 = slim.conv1d(X,num_features,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    re1 = Active(bn1,mode)
    x2 = slim.conv1d(re1,num_features,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    re2 = Active(bn2,mode)
    x3 = slim.conv1d(re2,num_features,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    res1 = slim.conv1d(X,num_features,1,stride=1,padding="SAME",activation_fn=None)
    res1 = batch_normal(res1,is_train)
    block1 = Active(bn3+res1,mode)
    return block1

def BaseBlock3(X,num_features,is_train):
    mode = 'relu'
    x1 = slim.conv1d(X,num_features,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    re1 = Active(bn1,mode)
    x2 = slim.conv1d(re1,num_features,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    re2 = Active(bn2,mode)
    x3 = slim.conv1d(re2,num_features,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    res1 = batch_normal(X,is_train)
    block1 = Active(bn3+res1,mode)
    return block1

###
###
def BuildResNetLSTMaN(X,num_label,length,rate=0.5,is_train=True):
    #####BLock 1
    num_features = 128
    features = []
    ########BLock2
    with tf.variable_scope("ResNet"):
            block1 = baseResidual(X,num_features,is_train)
            poo_1 = tf.layers.average_pooling1d(block1,length,1)
            poo_1 = slim.flatten(poo_1)
            avg_1 = slim.fully_connected(poo_1,num_label,activation_fn=None,weights_initializer=tf.glorot_uniform_initializer())
            features.append(avg_1)
            block2 = baseResidual(block1,num_features*2,is_train)
            poo_2 = tf.layers.average_pooling1d(block2,length,1)
            poo_2 = slim.flatten(poo_2)
            avg_2 = slim.fully_connected(poo_2,num_label,activation_fn=None,weights_initializer=tf.glorot_uniform_initializer())
            features.append(avg_2)
            block3 = BaseBlock3(block2,num_features*2,is_train)
            ful = tf.layers.average_pooling1d(block3,length,1)
            ful = slim.flatten(ful)
            avg_3 = slim.fully_connected(ful,num_label,activation_fn=None,weights_initializer=tf.glorot_uniform_initializer())
            features.append(avg_3)
    with tf.variable_scope("LSTM_ATTENTION1"):
            ls1 = LSTM_Attention(X,128)
            avg_4 = slim.flatten(ls1)
            avg_4 = slim.fully_connected(avg_4,num_label,activation_fn=None,weights_initializer=tf.glorot_uniform_initializer())
            features.append(avg_4)
    with tf.variable_scope("LSTM_ATTENTION2"):
            ls2 = LSTM_Attention(ls1,128)
            avg_5 = slim.flatten(ls2)
            avg_5 = slim.fully_connected(avg_5,num_label,activation_fn=None,weights_initializer=tf.glorot_uniform_initializer())
            features.append(avg_5)
    with tf.variable_scope("classifier",reuse=tf.AUTO_REUSE):
            ls2 = slim.flatten(ls2)
            ful = tf.concat([ful,ls2],axis=-1)
            #ful = tf.nn.dropout(ful,rate=rate)
            ful = slim.fully_connected(ful,num_label,activation_fn=None,weights_initializer=tf.glorot_uniform_initializer())
    return ful,features

def BuildResNet(X,num_label,length,rate=0.5,is_train=True):
    num_features = 128
    ########BLock2
    with tf.variable_scope("ResNet"):
            block1 = baseResidual(X,num_features,is_train)
            block2 = baseResidual(block1,num_features*2,is_train)
            block3 = BaseBlock3(block2,num_features*2,is_train)
            ful = tf.layers.average_pooling1d(block3,length,1)
            ful = slim.flatten(ful)
            ful = slim.fully_connected(ful,128,activation_fn=tf.nn.relu,weights_initializer=tf.glorot_uniform_initializer())
            ful = tf.nn.dropout(ful,rate=rate)
            ful = slim.fully_connected(ful,num_label,activation_fn=None,weights_initializer=tf.glorot_uniform_initializer())
    return ful


def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost


def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def super_soft_encross(predic,label):
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(label,predic))

##############################Parameters##################
def Totalcount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
        # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("The Total params:",total_parameters/1e6,"M")
    return total_parameters/1e6



################Define variants of ReLU

def CrossEntropyLossWithLabelSmoothing(y,logits,smooth_rate=0.001):
    with tf.variable_scope("LabelSmoothing"):
        losses = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, label_smoothing=smooth_rate)
        return losses



def ReLUVariants(x,choose=1):
    if choose == 1:
        x = tf.nn.relu(x)
    elif choose == 2:
        x = tf.nn.leaky_relu(x,alpha=0.1)
    elif choose == 3:
        x = prelu(x)
    elif choose == 4:
        x = tf.nn.sigmoid(x)
    else:
        x = tf.nn.tanh(x)
    return x

def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs



def ConvBLOCK(x,kernel,channel,is_train=True):
    x =slim.conv1d(x,channel,kernel,stride=1,padding="SAME",activation_fn=None)
    x = batch_normal(x,is_training=is_train)
    return x

def BasicBlock(x,channel,choose=1,is_train=True):
    x1 = ConvBLOCK(x,11,channel,is_train)
    #x1 = ReLUVariants(x1,choose)
    x2 = ConvBLOCK(x,17,channel,is_train)
    #x2 = ReLUVariants(x2,choose)
    x3 = tf.layers.max_pooling1d(x,2,1,padding="SAME")
    #x3 = ReLUVariants(x3,choose)
    x4 = tf.concat([x1,x2,x3],axis=-1)
    input_x = ConvBLOCK(x,1,x4.get_shape()[-1],is_train)
    result = x4 + input_x
    #result = ReLUVariants(result,choose)
    return result
##################Local Features#################
def PerceptiveLocalFeature(x,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("BasicBlock_1"):
        block_1_1 = BasicBlock(x,channel*2,choose,is_train)
        block_1 = ReLUVariants(block_1_1,choose)
        #block_1 = tf.layers.dropout(block_1,rate=0.2)
    with tf.variable_scope("BasicBlock_2"):
        block_2_1 = BasicBlock(block_1,channel*2,choose,is_train)
        block_2 = ReLUVariants(block_2_1,choose)
        #block_2 = tf.layers.dropout(block_2,rate=0.2)
    with tf.variable_scope("BasicBlock_3"):
        block_3 = BasicBlock(block_2,channel*2,choose,is_train)
    with tf.variable_scope("AVG_pooling"):
        avg_3 = attentd(ConvBLOCK(block_1,1,block_3.get_shape()[-1],is_train),ConvBLOCK(block_2,1,block_3.get_shape()[-1],is_train), block_3)
        avg_3 = avg_3 + ConvBLOCK(x,1,avg_3.get_shape()[-1],is_train)
        avg_3 = ReLUVariants(avg_3,choose)
        avg_3 = tf.layers.average_pooling1d(avg_3,length,1)
    with tf.variable_scope("FlattenShape"):
        avg = tf.layers.flatten(avg_3) 
    with tf.variable_scope("classifier"):
        avg = tf.layers.dropout(avg,rate=dropout)
        res = tf.layers.dense(avg,num_classes,activation=None)
        return res
####################Local Feature Extraction Without ExtensivePerception
def PerceptiveLocalFeatureWithoutExtensive(x,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("BasicBlock_1"):
        block_1_1 = BasicBlock(x,channel,choose,is_train)
        block_1 = ReLUVariants(block_1_1,choose)
        #block_1 = tf.layers.dropout(block_1,rate=0.2)
    with tf.variable_scope("BasicBlock_2"):
        block_2_1 = BasicBlock(block_1,channel*2,choose,is_train)
        block_2 = ReLUVariants(block_2_1,choose)
        #block_2 = tf.layers.dropout(block_2,rate=0.2)
    with tf.variable_scope("BasicBlock_3"):
        block_3 = BasicBlock(block_2,channel*2,choose,is_train)
    with tf.variable_scope("AVG_pooling"):
        avg_3 = tf.concat([block_1_1,block_2_1,block_3],axis=-1)
        avg_3 = avg_3 + ConvBLOCK(x,1,avg_3.get_shape()[-1],is_train)
        avg_3 = ReLUVariants(avg_3,choose)
        avg_3 = tf.layers.average_pooling1d(avg_3,length,1)
    with tf.variable_scope("FlattenShape"):
        avg = tf.layers.flatten(avg_3) 
    with tf.variable_scope("classifier"):
        avg = tf.layers.dropout(avg,rate=dropout)
        res = tf.layers.dense(avg,num_classes,activation=None)
        return res

def attentd(K,Q,V):
    att = tf.matmul(Q,tf.transpose(K,(0,2,1)))
    att = tf.nn.softmax(att,axis=-1)
    res = tf.matmul(att,V)
    return res


def MutualBlock(x,channel,drop,choose=1,is_train=True):
    x1 = ConvBLOCK(x,11,channel,is_train)
    x1 = ReLUVariants(x1,choose)
    x2 = ConvBLOCK(x,17,channel,is_train)
    x2 = ReLUVariants(x2,choose)
    x3 = ConvBLOCK(x,8,channel,is_train)
    x3 = ReLUVariants(x3,choose)
    result = attentd(x3,x1,x2)
    result = result +  ReLUVariants(ConvBLOCK(x,1,result.get_shape()[-1],is_train),choose)
    result = tf.layers.dropout(result,drop)
    return result


def PerceptiveMutualFeature(x,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("Mutual_block_1"):
        x1 = MutualBlock(x,channel,dropout,choose,is_train)
    with tf.variable_scope("Mutual_block_2"):
        x2 = MutualBlock(x1,channel,dropout,choose,is_train)
    with tf.variable_scope("Mutual_block_3"):
        x3 = MutualBlock(x2,channel,dropout,choose,is_train)
    with tf.variable_scope("Merge"):
        att = attentd(x1,x2,x3)
        att = att + ReLUVariants(ConvBLOCK(x,1,att.get_shape()[-1],is_train),choose)
        avg_3 = tf.layers.average_pooling1d(att,length,1)
    with tf.variable_scope("FlattenShape"):
        avg = tf.layers.flatten(avg_3) 
    with tf.variable_scope("classifier"):
        avg = tf.layers.dropout(avg,rate=dropout)
        res = tf.layers.dense(avg,num_classes,activation=None)
        return res

def baseResidualWith(X,num_features,drop,is_train):
    mode = 'relu'
    x1 = slim.conv1d(X,num_features,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    re1 = Active(bn1,mode)
    x2 = slim.conv1d(re1,num_features,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    re2 = Active(bn2,mode)
    x3 = slim.conv1d(re2,num_features,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    att = attentd(bn1,bn2,bn3)
    res1 = slim.conv1d(X,num_features,1,stride=1,padding="SAME",activation_fn=None)
    res1 = batch_normal(res1,is_train)
    block1 = Active(att+res1,mode)
    block1 = tf.layers.dropout(block1,rate=drop)
    return block1

def BaseBlock3With(X,num_features,drop,is_train):
    mode = 'relu'
    x1 = slim.conv1d(X,num_features,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    re1 = Active(bn1,mode)
    x2 = slim.conv1d(re1,num_features,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    re2 = Active(bn2,mode)
    x3 = slim.conv1d(re2,num_features,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    att = attentd(bn1,bn2,bn3)
    res1 = batch_normal(X,is_train)
    block1 = Active(att+res1,mode)
    block1 = tf.layers.dropout(block1,rate=drop)
    return block1

def BuildMutualResidualNet(X,channel,length,num_classes,dropout=0.5,is_train=True):
    num_features = channel
    ########BLock2
    with tf.variable_scope("MutualResNet"):
            block1 = baseResidualWith(X,num_features,dropout,is_train)
            block2 = baseResidualWith(block1,num_features,dropout,is_train)
            block3 = BaseBlock3With(block2,num_features,dropout,is_train)
            block3 = attentd(block1,block2,block3)
            ful = tf.layers.average_pooling1d(block3,length,1)
            ful = slim.flatten(ful)
            avg = tf.layers.dropout(ful,rate=dropout)
            res = tf.layers.dense(avg,num_classes,activation=None)
            return res


def CrossEntropyWithoutSmooth(Y,logit):
    with tf.variable_scope("scope"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(Y,logit),axis=-1)
    return loss





def Optimazer(loss,choose=1,lr=0.001,steps=50):
    global_step =  tf.Variable(0, name='global_step')
    lr = tf.train.exponential_decay(lr, global_step, steps, 0.9, staircase=True)
    if choose == 1:
        train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step)
    elif choose == 2 :
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss,global_step)
    else :
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step)
    return train_op,lr

def Top_1_Accuracy(label,predict):
    correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(predict,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

###############################################################
###############Transfer Model
###############################################################



def AttentionCapsuleScaleDot(Q,K,V,key_masks,dropout=0.0,training=True,scope="scaled_capsule_attention"):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        attend = tf.matmul(Q,tf.transpose(K,[0,2,1]))
        attend /= d_k ** 0.5
        attend = mask(attend,key_masks,type="f")
        #attend = tf.layers.dropout(attend,rate=dropout,training=training)
        attend = tf.nn.softmax(attend,-1)
        #attend = tf.sqrt(reduce_sum(tf.square(attend),axis=2,keepdims=True)+eposilion)
        pred = tf.matmul(attend,V)
        return pred

def MultiHeadAttention(queries,keys,values,key_masks,d_model=64,num_heads=8,dropout=0.0,training=True,scope="Multiattention"):
    #d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries,d_model,use_bias=True)
        K = tf.layers.dense(keys,d_model,use_bias=True)
        V = tf.layers.dense(values,d_model,use_bias=True)
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        outputs = AttentionCapsuleScaleDot(Q_,K_,V_,key_masks,dropout,training)
        outputs =tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
        outputs += tf.layers.dense(queries,d_model)
        outputs = ln(outputs)
        return outputs

def MultiHeadAttentionShared(queries,keys,values,key_masks,d_model=64,num_heads=8,dropout=0.0,training=True,scope="Multiattention"):
    #d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries,d_model//num_heads,use_bias=True)
        K = tf.layers.dense(keys,d_model,use_bias=True)
        V = tf.layers.dense(values,d_model,use_bias=True)
        #Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        Q_ = []
        for i in range(num_heads):
            Q_.append(Q)
        Q_ = tf.concat(Q_,axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        outputs = AttentionCapsuleScaleDot(Q_,K_,V_,key_masks,dropout,training)
        outputs =tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
        outputs += tf.layers.dense(queries,d_model)
        outputs = ln(outputs)
        return outputs



def mask(inputs,key_masks=None,type=None):
    padding_num = -2 ** 32 +1
    if type in ("k","key","keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks,[tf.shape(inputs)[0]//tf.shape(key_masks)[0],1])
        key_masks = tf.expand_dims(key_masks,1)
        output = inputs + key_masks * padding_num

    elif type in ("f","future","right"):
        diag_vals = tf.ones_like(inputs[0,:,:])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(inputs)[0],1,1])
        paddings = tf.ones_like( future_masks ) * padding_num
        output = tf.where(tf.equal(future_masks,0),paddings,inputs)
    else:
        print("Check if you entered type correctly")

    return output

def Dense(inputs,units):
    x = tf.layers.dense(inputs,units,use_bias=True)
    x = ln(x)
    return x

def BaseTransformer(inputs,key_masks,d_model=64,num_heads=8,choose=1,dropout=0.0,training=True,scope="BaseTransformer"):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        multihead = MultiHeadAttention(inputs,inputs,inputs,key_masks,d_model,num_heads,dropout,training)
        x1 = Dense(ReLUVariants(multihead,choose),d_model)
        x2 = Dense(ReLUVariants(x1,choose),d_model)
        outputs = ReLUVariants( x2+ multihead,choose  )
        return outputs


def TrainedMMLPaNLSTM(x,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("Pre-trainedTeacher"):
        with tf.variable_scope("PercepLocalFeature"):
            with tf.variable_scope("BasicBlock_1"):
                block_1_1 = BasicBlock(x,channel*2,choose,is_train)
                block_1 = ReLUVariants(block_1_1,choose)
                #block_1 = tf.layers.dropout(block_1,rate=0.2)
            with tf.variable_scope("BasicBlock_2"):
                block_2_1 = BasicBlock(block_1,channel*2,choose,is_train)
                block_2 = ReLUVariants(block_2_1,choose)
                #block_2 = tf.layers.dropout(block_2,rate=0.2)
            with tf.variable_scope("BasicBlock_3"):
                block_3 = BasicBlock(block_2,channel*2,choose,is_train)
            with tf.variable_scope("AVG_pooling"):
                avg_3 = attentd(ConvBLOCK(block_1,1,block_3.get_shape()[-1],is_train),ConvBLOCK(block_2,1,block_3.get_shape()[-1],is_train), block_3)
                avg_3 = avg_3 + ConvBLOCK(x,1,avg_3.get_shape()[-1],is_train)
                avg_3 = ReLUVariants(avg_3,choose)
                avg_3 = tf.layers.average_pooling1d(avg_3,length,1)
            with tf.variable_scope("FlattenShape"):
                avg = tf.layers.flatten(avg_3) 
        with tf.variable_scope("transformerCapsule"):
            with tf.variable_scope("LSTM_ATTENTION1"):
                    ls1 = LSTM_Attention(x,128)
            with tf.variable_scope("LSTM_ATTENTION2"):
                    ls2 = LSTM_Attention(ls1,128)
        with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(ls2)
            map_feature = tf.concat([avg,trans_capsules],axis=-1)
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result






Teacher_transformer_block_layers = 2
Teacher_transformer_perlayer = 72
Teacher_transformer_head = 8

def TransformerMTSC(X,num_label,length,drop=0.5,is_train=True):
    Teacher_transformer_block_layers = 4
    Teacher_transformer_perlayer = 64
    Teacher_transformer_head = 8
    choose=1
    results_list = []
    with tf.variable_scope("transformerCapsule"):
                trans_capsules = tf.transpose(X,(0,2,1))
                trans_mask = tf.math.equal(tf.transpose(X,(0,2,1)),0)
                with tf.variable_scope("transformers"):
                    for i in range(Teacher_transformer_block_layers):
                        with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                            trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                            num_heads=Teacher_transformer_head,choose=choose,dropout=drop,training=is_train)
                            results_list.append(trans_capsules)
    with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(trans_capsules)
            map_feature = trans_capsules
            results_list.append(map_feature)
            map_feature = tf.layers.dropout(map_feature,rate=drop)
            result = tf.layers.dense(map_feature,num_label,activation=None)
            return result,results_list









def TrainedTeacher(x,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("Pre-trainedTeacher"):
        with tf.variable_scope("PercepLocalFeature"):
            with tf.variable_scope("BasicBlock_1"):
                block_1_1 = BasicBlock(x,channel*2,choose,is_train)
                block_1 = ReLUVariants(block_1_1,choose)
                #block_1 = tf.layers.dropout(block_1,rate=0.2)
            with tf.variable_scope("BasicBlock_2"):
                block_2_1 = BasicBlock(block_1,channel*2,choose,is_train)
                block_2 = ReLUVariants(block_2_1,choose)
                #block_2 = tf.layers.dropout(block_2,rate=0.2)
            with tf.variable_scope("BasicBlock_3"):
                block_3 = BasicBlock(block_2,channel*2,choose,is_train)
            with tf.variable_scope("AVG_pooling"):
                avg_3 = attentd(ConvBLOCK(block_1,1,block_3.get_shape()[-1],is_train),ConvBLOCK(block_2,1,block_3.get_shape()[-1],is_train), block_3)
                avg_3 = avg_3 + ConvBLOCK(x,1,avg_3.get_shape()[-1],is_train)
                avg_3 = ReLUVariants(avg_3,choose)
                avg_3 = tf.layers.average_pooling1d(avg_3,length,1)
            with tf.variable_scope("FlattenShape"):
                avg = tf.layers.flatten(avg_3) 
        with tf.variable_scope("transformerCapsule"):
                trans_capsules = tf.transpose(x,(0,2,1))
                trans_mask = tf.math.equal(tf.transpose(x,(0,2,1)),0)
                with tf.variable_scope("transformers"):
                    for i in range(Teacher_transformer_block_layers):
                        with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                            trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                            num_heads=Teacher_transformer_head,choose=choose,dropout=dropout,training=is_train)
        with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(trans_capsules)
            map_feature = tf.concat([avg,trans_capsules],axis=-1)
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result
#######################################################
###################### InceptionTime
#######################################################
def BottleIncep(X,num_feature,is_train):
    x = slim.conv1d(X,num_feature,1,stride=1,padding="SAME",activation_fn=tf.nn.relu)
    kernles = [ 5, 7, 9,11,13]
    values = []
    for i in range(len(kernles)):
        values.append(slim.conv1d(x,num_feature,kernles[i],stride=1,padding="SAME",activation_fn=tf.nn.relu))
    conv1 = tf.layers.max_pooling1d(x,2,1,padding="SAME")
    conv2 = slim.conv1d(conv1,num_feature,1,stride=1,padding="SAME",activation_fn=tf.nn.relu)
    values.append(conv2)
    res = tf.concat(values,axis=-1)
    res = batch_normal(res,is_train)
    res_x = slim.conv1d(X,num_feature*4,1,stride=1,padding="SAME")
    res_x = batch_normal(res,is_train)
    res = res+res_x
    return Active(res)

def baseIncep(X,num_feature,is_train):
    x = X
    kernles = [5, 7, 9,11,13]
    values = []
    for i in range(len(kernles)):
        values.append(slim.conv1d(x,num_feature,kernles[i],stride=1,padding="SAME",activation_fn=tf.nn.relu))
    conv1 = tf.layers.max_pooling1d(x,2,1,padding="SAME")
    conv2 = slim.conv1d(conv1,num_feature,1,stride=1,padding="SAME",activation_fn=tf.nn.relu)
    values.append(conv2)
    res = tf.concat(values,axis=-1)
    res = batch_normal(res,is_train)
    res_x = slim.conv1d(X,num_feature*4,1,stride=1,padding="SAME")
    res_x = batch_normal(res,is_train)
    res = res+res_x
    return Active(res)

def InceptionTime(X,num_label,length,dropout=0.5,is_train=True):
    with tf.variable_scope("InceptionTime"):
        num_channel = 32
        bottle = BottleIncep(X,num_channel,is_train)
        depth = 5
        x = bottle
        for i in range(depth):
            x = baseIncep(x,num_channel,is_train)
        res = slim.conv1d(X,num_channel*6,1,1,padding="SAME",activation_fn=None)
        res = batch_normal(res,is_train)
        x = res+x
        x = Active(x)
        x = tf.layers.average_pooling1d(x,length,1)
        ful = slim.flatten(x)
        ful = tf.layers.dropout(ful,rate=dropout)
        ful = slim.fully_connected(ful,num_label,activation_fn=None)
        return ful


def TestResMuti(X,channel,length,num_classes,depth=1,choose=1,dropout=0.5,is_train=True):
    restore = []
    with tf.variable_scope("ResMuti"):
            num_channel = channel
            bottle = BottleIncep(X,num_channel,is_train)
            depth = depth
            x = bottle
            restore.append(x)
            for i in range(depth):
                x = baseIncep(x,num_channel,is_train)
                restore.append(x)
            res = slim.conv1d(X,num_channel* 6,1,1,padding="SAME",activation_fn=None)
            res = batch_normal(res,is_train)
            x = res+x
            x = Active(x)
            x = tf.layers.average_pooling1d(x,length,1)
            x = slim.flatten(x)
    with tf.variable_scope("classifier"):
            #trans_capsules = tf.layers.flatten(trans_capsules)
            map_feature = x
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result,restore




def TestTrans(X,channel,length,num_classes,depth=1,choose=1,dropout=0.5,is_train=True):
    restore = []
    with tf.variable_scope("transformerCapsule"):
                trans_capsules = tf.transpose(X,(0,2,1))
                trans_mask = tf.math.equal(tf.transpose(X,(0,2,1)),0)
                with tf.variable_scope("transformers"):
                    for i in range(Teacher_transformer_block_layers):
                        with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                            trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                            num_heads=Teacher_transformer_head,choose=choose,dropout=dropout,training=is_train)
    with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(trans_capsules)
            map_feature = trans_capsules
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result,restore

Teacher_transformer_block_layers = 3
Teacher_transformer_perlayer = 64
Teacher_transformer_head = 8

def DKN(X,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("Pre-trainedTeacher"):
        restore = []
        with tf.variable_scope("InceptionTime"):
            num_channel = channel
            bottle = BottleIncep(X,num_channel,is_train)
            depth = 4
            x = bottle
            restore.append(x)
            for i in range(depth):
                x = baseIncep(x,num_channel,is_train)
                restore.append(x)
            res = slim.conv1d(X,num_channel*6,1,1,padding="SAME",activation_fn=None)
            res = batch_normal(res,is_train)
            x = res+x
            x = Active(x)
            x = tf.layers.average_pooling1d(x,length,1)
            x = slim.flatten(x)
        with tf.variable_scope("transformerCapsule"):
                trans_capsules = tf.transpose(X,(0,2,1))
                trans_mask = tf.math.equal(tf.transpose(X,(0,2,1)),0)
                with tf.variable_scope("transformers"):
                    for i in range(Teacher_transformer_block_layers):
                        with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                            trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                            num_heads=Teacher_transformer_head,choose=choose,dropout=dropout,training=is_train)
                            restore.append(trans_capsules)
        with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(trans_capsules)
            map_feature = tf.concat([x,trans_capsules],axis=-1)
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result,restore




def AdaptiveClassifier(x,num_classes,drop):
    #with tf.variable_scope("classfier"):
        length = get_shape(x)[1]
        x = tf.layers.average_pooling1d(x,length,1)
        x =  slim.flatten(x)
        x = tf.layers.dropout(x,rate=drop)
        x = tf.layers.dense(x,num_classes,activation=None)
        return x


#############################################OtherNet###########################

def Self_Attention1D(x,chanel):
    query = slim.conv1d(x,chanel//2,1,1)
    key = slim.conv1d(x,chanel//2,1,1)
    value = slim.conv1d(x,chanel,1,1)
    b , w , c = get_shape(x)
    key = tf.reshape(key,[-1,chanel//2,w])
    attend = tf.matmul(query,key)
    attend = tf.nn.softmax(attend,axis=-1)
    out = tf.matmul(attend,value)
    #print(out)
    scale = tf.constant(1.0,tf.float32)
    out = scale * out +x
    return out

def ConvTime(X,num_label,is_train=True):
    x1 = tf.contrib.layers.conv1d(X,6,7,padding="SAME",activation_fn=tf.nn.sigmoid)
    x2 = tf.layers.AveragePooling1D(pool_size=3,strides=3)(x1)
    x3 = tf.contrib.layers.conv1d(x2,12,7,padding="SAME",activation_fn=tf.nn.sigmoid)
    x4 = tf.layers.AveragePooling1D(pool_size=3,strides=3)(x3)
    x5 = tf.layers.flatten(x4)
    y = tf.contrib.layers.fully_connected(x5,num_label,activation_fn=None)
    return y

def EncodeModel(X,num_label,is_train=True):
    ##### first-1
    x1 = tf.contrib.layers.conv1d(X,128,5,stride=1,padding='SAME',activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    rel = prelu(bn1)
    rel = tf.nn.dropout(rel,0.2)
    max1 = tf.layers.max_pooling1d(rel,2,2,padding='SAME') 
    x2 = tf.contrib.layers.conv1d(max1,256,11,stride=1,padding='SAME',activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    rel = prelu(bn2)
    rel = tf.nn.dropout(rel,0.2)
    max2 = tf.layers.max_pooling1d(rel,2,2,padding='SAME') 
    x3 = tf.contrib.layers.conv1d(max2,512,21,stride=1,padding='SAME',activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    rel = prelu(bn3)
    rel = tf.nn.dropout(rel,0.2)
    max3 = tf.layers.max_pooling1d(rel,2,2,padding='SAME') 
    max3 = batch_normal(max3,is_train)
    ##   Attention Mechanism
    ###
    attent = Self_Attention1D(max3,512)
    ful_feature = tf.layers.flatten(attent)
    soft = tf.contrib.layers.fully_connected(ful_feature,num_label,activation_fn=tf.sigmoid)
    return soft

def BuildFCN(X,num_label,length,drop=0.5,is_train=True):
    mode = 'relu'
    results = []
    x1 = tf.contrib.layers.conv1d(X,128,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    ac1 = Active(bn1,mode)
    results.append(ac1)
    x2 = tf.contrib.layers.conv1d(ac1,256,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    ac2 = Active(bn2,mode)
    results.append(ac2)
    x3 = tf.contrib.layers.conv1d(ac2,128,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    ac3 = Active(bn3,mode)
    results.append(ac3)
    avg = tf.layers.average_pooling1d(ac3,length,1)
    fc = tf.layers.flatten(avg)
    fc = tf.contrib.layers.fully_connected(fc,128,activation_fn=tf.nn.relu)
    fc = tf.nn.dropout(fc,drop)
    results.append(fc)
    fc = tf.contrib.layers.fully_connected(fc,num_label,activation_fn=None)
    return fc,results

def FCNLSTMN(X,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("Pre-trainedStudent"):
        with tf.variable_scope("FCNwith"):
            x1 = ConvBLOCK(X,8,channel,is_train=is_train)
            x1 = ReLUVariants(x1,choose)
            x2 = ConvBLOCK(x1,5,channel*2,is_train=is_train)
            x2 = ReLUVariants(x2,choose)
            x3 = ConvBLOCK(x2,3,channel,is_train=is_train)
            x3 = ReLUVariants(x3,choose)
            x4 = tf.layers.average_pooling1d(x3,length,1)
            avg = slim.flatten(x4)
        with tf.variable_scope("transformerCapsule"):
                trans_capsules = tf.transpose(X,(0,2,1))
                trans_mask = tf.math.equal(tf.transpose(X,(0,2,1)),0)
                with tf.variable_scope("transformers"):
                    for i in range(Teacher_transformer_block_layers):
                        with tf.variable_scope("num_block_{}".format(i+1),reuse=tf.AUTO_REUSE):
                            trans_capsules = BaseTransformer(trans_capsules,trans_mask,d_model=int((i+1)*Teacher_transformer_perlayer),
                            num_heads=Teacher_transformer_head,choose=choose,dropout=dropout,training=is_train)
        with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(trans_capsules)
            map_feature = tf.concat([avg,trans_capsules],axis=-1)
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result

def FCNTransformer(X,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("Pre-trainedStudent"):
        with tf.variable_scope("FCNwith"):
            x1 = ConvBLOCK(X,8,channel,is_train=is_train)
            x1 = ReLUVariants(x1,choose)
            x2 = ConvBLOCK(x1,5,channel,is_train=is_train)
            x2 = ReLUVariants(x2,choose)
            x3 = ConvBLOCK(x2,3,channel,is_train=is_train)
            x3 = ReLUVariants(x3,choose)
            x4 = tf.layers.average_pooling1d(x3,length,1)
            avg = slim.flatten(x4)
        with tf.variable_scope("transformerCapsule"):
            with tf.variable_scope("LSTM_ATTENTION1"):
                    ls1 = LSTM_Attention( tf.transpose(X,(0,2,1)),channel//4)
            with tf.variable_scope("LSTM_ATTENTION2"):
                    ls2 = LSTM(ls1,channel//2)[-1]
        with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(ls2)
            map_feature = tf.concat([avg,trans_capsules],axis=-1)
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result


def layers(X):
    x1 = ConvBLOCK(X,3,500)
    x1 = ReLUVariants(x1,1)
    x2 =  ConvBLOCK(x1,3,300)
    x2 = ReLUVariants(x2,1)
    return x2

def TapNet(X,length,num_classes,choose=1,dropout=0.5,is_train=True):
    channel = 256
    with tf.variable_scope("multi-head"):
            x1 = ConvBLOCK(X,8,channel,is_train=is_train)
            x1 = ReLUVariants(x1,choose)
            x2 = ConvBLOCK(x1,5,channel,is_train=is_train)
            x2 = ReLUVariants(x2,choose)
            x3 = ConvBLOCK(x2,3,channel//2,is_train=is_train)
            x3 = ReLUVariants(x3,choose)
    with tf.variable_scope("shared_layers",reuse=tf.AUTO_REUSE):
            x1 = layers(x1)
            x2 = layers(x2)
            x3 = layers(x3)
    with tf.variable_scope("avg_"):
            x1 = tf.layers.average_pooling1d(x1,length,1)
            x1 = slim.flatten(x1)
            x2 = tf.layers.average_pooling1d(x2,length,1)
            x2 = slim.flatten(x2)
            x3 = tf.layers.average_pooling1d(x3,length,1)
            x3 = slim.flatten(x3)
    with tf.variable_scope("LSTRTMM"):
            ls2 = LSTM(X,channel//2)[-1]
    with tf.variable_scope("classifier"):
            trans_capsules = tf.layers.flatten(ls2)
            map_feature = tf.concat([x1,x2,x3,trans_capsules],axis=-1)
            map_feature = tf.layers.dropout(map_feature,rate=dropout)
            #map_feature = tf.layers.dense(map_feature,128)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
    return result            













def FCNEx(X,channel,length,num_classes,choose=1,dropout=0.5,is_train=True):
     with tf.variable_scope("Pre-trainedStudent"):
        with tf.variable_scope("FCNwith"):
            x1 = ConvBLOCK(X,8,channel,is_train=is_train)
            x1 = ReLUVariants(x1,choose)
            x2 = ConvBLOCK(x1,8,channel*2,is_train=is_train)
            x2 = ReLUVariants(x2,choose)
            x3 = ConvBLOCK(x2,8,channel,is_train=is_train)
            x3 = ReLUVariants(x3,choose)
            x4 = tf.layers.average_pooling1d(x3,length,1)
            avg = slim.flatten(x4)
        with tf.variable_scope("classifier"):
            #trans_capsules = tf.layers.flatten(ls2)
            #map_feature = tf.concat([avg,trans_capsules],axis=-1)
            map_feature = tf.layers.dropout(avg,rate=dropout)
            result = tf.layers.dense(map_feature,num_classes,activation=None)
            return result

def kl_divergence_with_logits(p_logits, q_logits):
        p = p_logits
        log_p = tf.log(p_logits)
        log_q = tf.log(q_logits)
        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return kl

def soft_kl_divergence_with_logits(p_logits, q_logits):
        p = tf.nn.softmax(p_logits)
        log_p = tf.nn.log_softmax(p_logits)
        log_q = tf.nn.log_softmax(q_logits)

        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return tf.reduce_mean(kl)


def cross_entropy_loss(p,q):
    ce_loss =  tf.reduce_sum(- p*tf.log(q), -1)
    return ce_loss

def L1_loss(p,q):
    dif = tf.abs(p-q)
    l1_loss = tf.reduce_sum(dif,axis=-1)
    return l1_loss

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.reduce_sum(tf.select(condition, small_res, large_res),axis=-1)


def L2_loss(teacher,student):
    square =tf.square(teacher-student)
    square = tf.reduce_sum(square,axis=-1)
    return tf.reduce_mean(square)


def FCNwithSelfDistiilation(X,Y,channel,length,num_classes,coffie=0.1,choose=1,dropout=0.5,is_train=True):
    with tf.variable_scope("Pre-trainedStudent"):
        with tf.variable_scope("FCNwith"):
            x1 = ConvBLOCK(X,8,channel,is_train=is_train)
            x1 = ReLUVariants(x1,choose)
            x1_pool = tf.layers.average_pooling1d(x1,length,1)
            x1_pool = slim.flatten(x1_pool)
            x1_avg =  tf.layers.dense(x1_pool,num_classes,activation=None)

            x2 = ConvBLOCK(x1,8,channel*2,is_train=is_train)
            x2 = ReLUVariants(x2,choose)
            x2_pool = tf.layers.average_pooling1d(x2,length,1)
            x2_pool = slim.flatten(x2_pool)
            x2_avg =  tf.layers.dense(x2_pool,num_classes,activation=None)
            
            x3 = ConvBLOCK(x2,8,channel,is_train=is_train)
            x3 = ReLUVariants(x3,choose)
            x4 = tf.layers.average_pooling1d(x3,length,1)
            avg = slim.flatten(x4)
        with tf.variable_scope("classifier"):
            #trans_capsules = tf.layers.flatten(ls2)
            #map_feature = tf.concat([avg,trans_capsules],axis=-1)
            #map_feature = tf.layers.dropout(avg,rate=dropout)
            result = tf.layers.dense(avg,num_classes,activation=None)
        with tf.variable_scope("WithLosses"):
            supervise_loss = CrossEntropyWithoutSmooth(Y,result)
            distill_losses = L1_loss(result,x1_avg) + L1_loss(result,x2_avg)
            losses = tf.reduce_mean((1-coffie)*supervise_loss + coffie * distill_losses,axis=-1)
            return losses,result









def BuildModelMCC(X,num_label,is_train=True):
    x = tf.contrib.layers.conv1d(X,8,5,1,padding="SAME",activation_fn=tf.nn.relu)
    x = tf.layers.max_pooling1d(x,2,2,padding="SAME")
    x = tf.contrib.layers.conv1d(x,8,5,stride=1,padding="SAME",activation_fn=tf.nn.relu)
    x = tf.layers.max_pooling1d(x,2,2,padding="SAME")
    x = tf.layers.flatten(x)
    fully = slim.fully_connected(x,732,activation_fn=tf.nn.relu)
    out = slim.fully_connected(fully,num_label,activation_fn=None)
    return out

def BuildMLP(X,num_label,is_train=True):
    x = slim.flatten(X)
    x = slim.dropout(x,0.1)
    x = slim.fully_connected(x,500,activation_fn=tf.nn.relu)
    x = slim.dropout(x,0.2)
    x = slim.fully_connected(x,500,activation_fn=tf.nn.relu)
    x = slim.dropout(x,0.2)
    x = slim.fully_connected(x,500,activation_fn=tf.nn.relu)
    x = slim.dropout(x,0.3)
    x = slim.fully_connected(x,num_label,activation_fn=None)
    return x

