#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
import time

batch_size=32
console_ = True
CPU_limit = False

if(CPU_limit):
  config = tf.ConfigProto(device_count={'CPU': 1})
  sess = tf.InteractiveSession(config=config)
else:
  sess = tf.InteractiveSession()





def next_batch(features,label,idx_count):
  if(idx_count+batch_size>features.shape[0]):
    batch_img = features[idx_count:,:]
    batch_label = label[idx_count:,:]
    idx_count = 0
  else:
    batch_img = features[idx_count:idx_count+batch_size,:]
    batch_label = label[idx_count:idx_count+batch_size,:]
    idx_count = idx_count+batch_size
  return batch_img, batch_label,idx_count
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  #with tf.device('/cpu:0'):
  type = tf.float32
  var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def copa(name,images,conv_shape,conv_srtride,pool_size,pool_stride,activation):
  kernel = _variable_with_weight_decay(name+'_weights',
                                   shape=conv_shape,
                                   stddev=5e-2,
                                   wd=0.0)
  with tf.device('/gpu:0'):
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
  biases = _variable_on_cpu(name+'_biases', conv_shape[3], tf.constant_initializer(0.0))
  pre_activation = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(pre_activation)
  #_activation_summary(conv)
  # pooling
  pool1 = tf.nn.max_pool(conv1, ksize=pool_size, strides=pool_stride,
                         padding='SAME')
  # normalization
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name=name+'_norm')
  return norm1

def linpa(name,features,out_dim):
  # Move everything into depth so we can perform a single matrix multiply.
  in_dim = features.get_shape()[1]
  weights = _variable_with_weight_decay(name+'_weights', shape=[in_dim, out_dim],
                                        stddev=0.04, wd=0.004)
  biases = _variable_on_cpu(name+'_biases', [out_dim], tf.constant_initializer(0.1))
  local = tf.nn.relu(tf.matmul(features, weights) + biases)
  return local

def softlin(name,features,out_dim):
  in_dim = features.shape[1]
  weights = _variable_with_weight_decay(name+'_weights', [in_dim, out_dim],
                                        stddev=1/in_dim, wd=0.0)
  biases = _variable_on_cpu(name+'_biases', [out_dim],
                            tf.constant_initializer(0.0))
  softmax_linear = tf.add(tf.matmul(features, weights), biases)
  #_activation_summary(softmax_linear)
  return softmax_linear

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

max_step = 10000000

x = tf.placeholder(tf.float32, shape=[None, 4096])
#x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 9])

##First Convolutional layer
#W_conv1 = weight_variable([5, 5, 1, 32])
#b_conv1 = bias_variable([32])
#
##Second convolutional layer
#W_conv2 = weight_variable([5, 5, 32, 64])
#b_conv2 = bias_variable([64])
#
##First fully connected layer
#W_fc1 = weight_variable([7 * 7 * 64, 1024])
#b_fc1 = bias_variable([1024])
#
##Second fully connected
#W_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])
#
#x_image = tf.reshape(x, [-1,28,28,1])
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)
#
#
#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#
#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

idx_count = 0
execfile('mat2py.py')
train_img = fvec_shuffle[:1000,:]
train_img = train_img/np.amax(train_img)
train_label_oneHot = label_oneHot_shuffle[:1000,:]
train_label = label_shuffle[:1000]
test_img = fvec_shuffle[1000:,:]
test_img = test_img/np.amax(train_img)
test_label_oneHot = label_oneHot_shuffle[1000:,:]
test_label = label_shuffle[1000:]

#h_conv1 = copa('conv1',x_image,[5, 5, 3, 64],[1,1,1,1],[1,3,3,1],[1,2,2,1],0)
#h_conv2 = copa('conv2',h_conv1,[5, 5, 64,64],[1,1,1,1],[1,3,3,1],[1,2,2,1],0)
#h_conv2_flat = tf.reshape(h_conv2, [-1, 4096])
#h_fc1 = linpa('fc1',x,4096)
h_fc2 = linpa('fc2',x,1024)
keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
h_soft = linpa('softmax',h_fc2_drop,9)

y_conv=h_soft

#Train
#cross_entropy = tf.cast(tf.reduce_mean(tf.square(tf.sub(tf.argmax(y_,1), tf.argmax(y_conv,1)))),tf.float32)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
cir1_prediction = tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)-1),tf.float32) +\
tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)),tf.float32) +\
tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)+1),tf.float32)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cir1_accuracy = tf.reduce_mean(tf.cast(cir1_prediction, tf.float32))
#accuracy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#      labels=tf.cast(correct_prediction, tf.int64), logits=y_, name='cross_entropy_per_example'))
sess.run(tf.global_variables_initializer())

duration = 0.
time_begin = time.time()

for i in range(max_step):
  batch_img, batch_label,idx_count = next_batch(train_img,train_label_oneHot,idx_count)
  if i%100 == 0:
    f = open('cifar_CNN_train.out', 'a')
    train_accuracy = accuracy.eval(feed_dict={x:batch_img, y_: batch_label, keep_prob: 1.0})
    train_cir1 = cir1_accuracy.eval(feed_dict={x:batch_img, y_: batch_label, keep_prob: 1.0})
    train_cross_entropy = cross_entropy.eval(feed_dict={x:batch_img, y_: batch_label, keep_prob: 1.0})
    if(console_):
      print("step %d, idx %d, training accuracy %g,CIR-1 %g, cross entropy is %g(%.3f sec/batch=%d images)"
      %(i,  idx_count,     train_accuracy,train_cir1,train_cross_entropy,duration,batch_size))
      y_batch = np.argmax(y_conv.eval(feed_dict={x:batch_img, y_: batch_label, keep_prob: 1.0}),axis=1)
      #print(y_batch)
    else:
      f.write("step %d, idx %d, training accuracy %CIR-1 %g,cross entropy is %g(%.3f sec/batch=%d images) \n"
      %(i,  idx_count,     train_accuracy,train_cir1,train_cross_entropy,duration,batch_size))
  if i%1000 == 0:
    f = open('cifar_CNN_test.out', 'a')
    test_accuracy = accuracy.eval(feed_dict={x:test_img[:1500,:], y_: test_label_oneHot[:1500,:], keep_prob: 1.0})
    test_cir1 = cir1_accuracy.eval(feed_dict={x:test_img[:1500,:], y_: test_label_oneHot[:1500,:], keep_prob: 1.0})
    test_cross_entropy = cross_entropy.eval(feed_dict={x:test_img[:1500,:], y_: test_label_oneHot[:1500,:], keep_prob: 1.0})
    if(console_):
      print("STEP %d, TEST ACCURACY %g,CIR-1 %g, CROSS ENTROPY %g"%(i, test_accuracy,test_cir1,test_cross_entropy))
    else:
      f.write("STEP %d, TEST ACCURACY %g,CIR-1 %g,CROSS ENTROPY %g \n"%(i, test_accuracy,test_cir1,test_cross_entropy))
  time_start = time.time()
  train_step.run(feed_dict={x: batch_img, y_: batch_label, keep_prob: 1.0})
  duration = time.time()-time_start

duration_total=time.time()-time_begin
print("Total time is %.3f sec"%duration_total)



