import tensorflow as tf
import numpy as np
import os
import pickle
import reinforce2
import threading
import time

LEFT  = np.array([0,-1])
RIGHT = np.array([0,1])
UP    = np.array([1,0])
DOWN  = np.array([-1,0])
DEVICE = '/cpu:0'


board_size = 21
patch_sizes= [8,4,4]
depths = [32,64,64]
strides = [4,2,1]
num_hidden1 = 512
num_hidden2 = 256
num_labels = 1
board_size = board_size
batch_size = 50
learn_rate = 0.005
L2_penalty = 0.001
num_epochs = 100
#N = 2000

valid_boards = None
valid_labels = None


######

filenames = ['data/board_outcome_'+str(i)+'.tfrecords' for i in range(12)]
filename_queue = tf.train.string_input_producer(string_tensor = filenames,
	num_epochs = num_epochs,shuffle = True)


def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
		'boards': tf.FixedLenFeature([board_size,board_size], tf.float32),
		'results': tf.FixedLenFeature([1], tf.float32),
		},
		name= 'board_res')
	print features
	#board = tf.decode_raw(features['board_raw'],tf.float32)
	#result = tf.decode_raw(features['result_raw'],tf.float32)
	board = features['boards']
	result = features['results']
	board = tf.reshape(board,(board_size,board_size,1))
	#result = tf.cast(result,tf.float32)
	result.set_shape([1])
	return board,result

board, result = read_and_decode(filename_queue)

board_batch, result_batch = tf.train.shuffle_batch([board,result],
	batch_size = batch_size, num_threads = 2, capacity = 10000 + 3*batch_size,
	min_after_dequeue = 10000)

print board_batch, result_batch
######

def model(cells):
	with tf.device(DEVICE):
		conv = tf.nn.conv2d(cells, w1, [1, 4, 4, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b1)
		conv = tf.nn.conv2d(hidden, w2, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b2)
		conv = tf.nn.conv2d(hidden, w3, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b3)

		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden1 = tf.nn.relu(tf.matmul(reshape, w4) + b4)
		hidden2 = tf.nn.relu(tf.matmul(hidden1, w5) + b5)
		out = tf.nn.sigmoid(tf.matmul(hidden2, w6) + b6)
	return out

def model_train(cells):
	with tf.device(DEVICE):
		conv = tf.nn.conv2d(cells, w1, [1, 4, 4, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b1)
		conv = tf.nn.conv2d(hidden, w2, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b2)
		conv = tf.nn.conv2d(hidden, w3, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b3)

		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden1 = tf.nn.relu(tf.matmul(reshape, w4) + b4)
		hidden_dropped1 = tf.nn.dropout(hidden1,0.5)
		hidden2 = tf.nn.relu(tf.matmul(hidden_dropped1, w5) + b5)
		hidden_dropped2 = tf.nn.dropout(hidden2,0.5)

		out = tf.nn.sigmoid(tf.matmul(hidden_dropped2, w6) + b6)
	return out





# valid_boards = tf.constant(valid_boards)
# valid_labels = tf.constant(valid_labels)

# define weights, biases of each layer
w1 = tf.Variable(tf.truncated_normal(
	[patch_sizes[0], patch_sizes[0], 1, depths[0]], stddev=0.1))
b1 = tf.Variable(tf.zeros([depths[0]]))

w2 = tf.Variable(tf.truncated_normal(
	[patch_sizes[1], patch_sizes[1], depths[0], depths[1]], stddev=0.01))
b2 = tf.Variable(tf.constant(1.0,shape=[depths[1]]))

w3 = tf.Variable(tf.truncated_normal(
	[patch_sizes[2], patch_sizes[2], depths[1], depths[2]], stddev=0.001))
b3 = tf.Variable(tf.constant(1.0,shape=[depths[2]]))

w4 = tf.Variable(tf.truncated_normal(
	# [board_size // 4 * board_size // 4 * depth, num_hidden], stddev=0.1))
	[6*6*16, num_hidden1], stddev=0.001))
b4 = tf.Variable(tf.constant(1.0,shape=[num_hidden1]))

w5 = tf.Variable(tf.truncated_normal(
	[num_hidden1, num_hidden2], stddev=0.001))
b5 = tf.Variable(tf.constant(1.0,shape=[num_hidden2]))

w6 = tf.Variable(tf.truncated_normal(
	[num_hidden2, num_labels], stddev=0.001))
b6 = tf.Variable(tf.constant(1.0,shape=[num_labels]))

regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + \
	 tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + \
	 tf.nn.l2_loss(w5) +tf.nn.l2_loss(w6)


logits = model_train(board_batch)
# loss1 = tf.nn.softmax_cross_entropy_with_logits(
# 	labels=board_result_batch[1],
# 	logits=logits)

loss1 = tf.reduce_mean(tf.nn.l2_loss(logits - result_batch))

loss = tf.reduce_mean(loss1+L2_penalty*regularizers)

if valid_boards is not None:
	valid_logits = model(valid_boards)
	valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		labels=valid_labels,logits=valid_logits))

#optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)			
optimizer = tf.train.MomentumOptimizer(learn_rate,0.9).minimize(loss)			
# qr = tf.train.QueueRunner(queue,[enqueue_op]*4)

tf.summary.scalar('train_loss',loss)
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./checkpoints')

init_op = tf.group(tf.global_variables_initializer(),
	tf.local_variables_initializer())
with tf.Session() as sess:
	coord = tf.train.Coordinator()
	sess.run(init_op)

	threads = tf.train.start_queue_runners(sess=sess,coord=coord)

	print 'Step:','Train_loss'
	n = 0
	loss_list = []
	try:
#		for n in range(N):
#			if coord.should_stop():
#				break
		while not coord.should_stop():
			n += 1
			# b,r = sess.run([board,result])
			# print b,r


			_ ,l,summary= sess.run([optimizer,loss,merged])
			summary_writer.add_summary(summary,n)
			loss_list.append(l)
			if n % 100 == 0:
				l_mean = np.mean(loss_list)
				print n, l_mean, time.time()
				
				loss_list = []


	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	coord.request_stop()
	coord.join(threads)



def save_session(id):
	save_path = saver.save(sess,"./models/model"+str(id)+".ckpt")
	print("Model saved in file: %s" % save_path)
	pass

def manual_restore(restore_id):
	print 'restoring from: ',"./models/model"+str(restore_id)+".ckpt"
	saver.restore(sess,"./models/model"+str(restore_id)+".ckpt")
	pass





