#!/usr/bin/python
# -*- coding: UTF-8 -*-
#fuser 6007/tcp -k
import numpy as np
import os
import tensorflow as tf
import math
import pickle
import random
from tensorflow.contrib import rnn
###### Do not modify here ###### 

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    

labelspace = [chr(i+ord('a')) for i in range(26)]

def labelEncode(char):
    r = [0 for c in labelspace]
    r[labelspace.index(char)] = 1
    return r
    
def labelDecoder(arr):
    return arr.index(max(arr))
    
def list2String(arr):
    r = "[ "
    for i in range(len(arr)):
        r += str(arr[i])
        if i < len(arr)-1:
            r += ", "
    return r
reset_graph()

train_set = pickle.load(open("train_set.dat", "rb"))
test_set = pickle.load(open("validate_set.dat", "rb"))

print("all labels: ", end="") 
print(train_set.keys())

# training on data but only on alpha a to z
axiss = ['ay', 'az', 'ray', 'raz', 'gx', 'rgx', 'gy', 'rgy', 'gz', 'rgz', 'gs', 'rgs']
X_train1 = []
y_train1 = []

X_valid1 = []
y_valid1 = []

X_test1 = []
y_test1 = []

# shuffle trainninf set
items = []
for label, values in train_set.items():
    for v in values:
        items.append((label, v))
test_items = []
for label, values in test_set.items():
    for v in values:
        test_items.append((label, v))
            
shuffled = sorted(items, key=lambda k: random.random())

for label, data in shuffled:    
    
    # rebuild data structure
    img = []
    width = len(data['ax'])
    for i in range(width):
        img += [data[axis][i] for axis in axiss]
    
    X_train1.append(img)
    y_train1.append(labelEncode(label))
    
for label, data in test_items:
    img = []
    width = len(data['ax'])
    for i in range(width):
        img += [data[axis][i] for axis in axiss]
    X_test1.append(img)
    y_test1.append(labelEncode(label))
        
print('dataset loading has finished')
###### Do not modify here ###### 

learning_rate = 0.001       # learning rate
epochs = 2500                 # epochs
batch_size = 512          # batch size
num_inputs = len(axiss)             # MNIST data input (img shape: len(axiss)*100)
num_steps = len(data[axiss[0]])               # time steps
num_neurons = 64      	 # neurons in hidden layer
num_classes = len(labelspace)             

X_train1 = np.asarray(X_train1)
X_train1 = X_train1.reshape([-1, num_steps, num_inputs])

X_test1 = np.asarray(X_test1)
X_test1 = X_test1.reshape([-1, num_steps, num_inputs])

x = tf.placeholder(tf.float32, [None, num_steps, num_inputs], name = 'x')
y = tf.placeholder(tf.float32, [None, num_classes], name = 'labels')

with tf.name_scope("init_weights"):
	weights = {
		# shape (N, 64)
		'in': tf.Variable(tf.random_normal([num_inputs, num_neurons]), name = "weights_in"),
		# shape (64, 26)
		'out': tf.Variable(tf.random_normal([num_neurons, num_classes]), name = "weight_out")
	}

with tf.name_scope("init_biases"):
	biases = {
		# shape (128, )
		'in': tf.Variable(tf.constant(0.1, shape=[num_neurons, ]), name = "biases_in"),
		# shape (5, )
		'out': tf.Variable(tf.constant(0.1, shape=[num_classes, ]), name = "biases_out")
	}

def RNN(X, weights, biases, name = "RNN"):
	with tf.name_scope("input_layer"):
		X = tf.reshape(X, [-1, num_inputs])
		X_in = tf.matmul(X, weights['in']) + biases['in']
		#LSTM accept input shape is [batch_size, time_step, num_neurons]
		X_in = tf.reshape(X_in, [-1, num_steps, num_neurons])
	with tf.name_scope("RNN_CELL"):
		cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, forget_bias=1.0, state_is_tuple=True)
		#states[0]= c(t), states[1]=h(t)
		outputs, states = tf.nn.dynamic_rnn(cell, X_in, dtype=tf.float32)
	with tf.name_scope('output_layer'):
		#results = h(t)*V
		results = tf.matmul(states[1], weights['out']) + biases['out']
	return results
output = RNN(x, weights, biases)

# weights = {
#     # Hidden layer weights => 2*n_hidden because of forward + backward cells
#     'out': tf.Variable(tf.random_normal([2*num_neurons, num_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([num_classes]))
# }


# def BiRNN(X, weights, biases):

#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, timesteps, n_input)
#     # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

#     # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
#     X = tf.unstack(X, num_steps, 1)

#     # Define lstm cells with tensorflow
#     # Forward direction cell
#     lstm_fw_cell = rnn.BasicLSTMCell(num_neurons, forget_bias=1.0)
#     # Backward direction cell
#     lstm_bw_cell = rnn.BasicLSTMCell(num_neurons, forget_bias=1.0)

#     # Get lstm cell output
#     try:
#         outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X,
#                                               dtype=tf.float32)
#     except Exception: # Old TensorFlow version only returns outputs not states
#         outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X,
#                                         dtype=tf.float32)

#     # Linear activation, using rnn inner loop last output
#     print(outputs[-1])
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']

# output = BiRNN(x, weights, biases)


with tf.name_scope("softmax_layer"):
	pred = tf.nn.softmax(output,name="prediction")

with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred , labels=y), name = "loss")

with tf.name_scope("train"):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, name = "train_step")

with tf.name_scope("accuracy"):
	correct_pred = tf.equal(tf.argmax(pred, 1, name = "prediction_matrix"), tf.argmax(y, 1, name = "label_matrix"), name = "correct_pred")
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")

init = tf.global_variables_initializer()

import time
arr = []
st = time.time()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
max_accuracy = 0.0
with tf.Session(config=config) as sess:
	writer = tf.summary.FileWriter("logs/", sess.graph)
	sess.run(init)
	for epoch in range(epochs):
		i = 0
		j = batch_size
		for k in range(int(math.ceil(len(X_train1)/batch_size))):
			epoch_x, epoch_y= X_train1[i:j], y_train1[i:j]
			sess.run(train_step, feed_dict = {x:epoch_x, y:epoch_y})
			i = i + batch_size
			j = j + batch_size
            
		result = sess.run(accuracy, feed_dict={x:X_test1, y:y_test1})
		if result > max_accuracy: 
			max_accuracy = result
			save_path = saver.save(sess, "./RNNLevel1")
            
		arr.append(result)
		print("Epoch:%d, Accuracy:%lf" %((epoch+1), result))
            
	end = time.time()
    
	print('training finish.\ncost time:',int(end-st))
	print(sess.run(weights['out']))
    
	result_matrix = [[0 for i in range(len(labelspace))] for j in range(len(labelspace))]
	results = sess.run(pred, feed_dict={x:X_test1})
	for i, result in enumerate(results):
# 		print(result)
		pred_label = labelDecoder(result.tolist())
		real_label = labelDecoder(y_test1[i])
		result_matrix[real_label][pred_label] += 1
# 		print('pred: %d(%s), real: %d(%s)' % (pred_label, labelspace[pred_label], real_label, labelspace[real_label]))
# 		print("==============================================================================")
	for i, raw in enumerate(result_matrix):
		print(labelspace[i]+'\t', end="")
		print(raw)
	pickle.dump(result_matrix, open('result.mtx', 'wb'))
	print(labelspace)
        
	fp = open('record', 'w')
	fp.write("labelspace = "+list2String(labelspace)+'], num_neurons = '+str(num_neurons)+'\n')
	for num in arr:
		fp.write(str(num)+'\n')
	fp.close()