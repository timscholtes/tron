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
num_hidden1 = 1024
num_hidden2 = 512
num_hidden3 = 256
num_hidden4 = 128
num_labels = 1
board_size = board_size
batch_size = 4
learn_rate = 0.01
L2_penalty = 0
num_epochs = 200
num_threads = 2
print_freq = 500
#N = 2000

valid_boards = None
valid_labels = None


######

filenames = ['data3/board_outcome_'+str(i)+'.tfrecords' for i in range(8)]
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
	#board = tf.reshape(board,(board_size,board_size,1))
	board = tf.reshape(board,(board_size**2,))
	#result = tf.cast(result,tf.float32)
	result.set_shape([1])
	return board,result

board, result = read_and_decode(filename_queue)

board_batch, result_batch = tf.train.shuffle_batch([board,result],
	batch_size = batch_size, num_threads = num_threads, capacity = 10000 + 3*batch_size,
	min_after_dequeue = 10000)

print board_batch, result_batch
######

def model(cells,activation):
	with tf.device(DEVICE):
		layer1 = activation(tf.matmul(cells,w1)+b1)
		layer2 = activation(tf.matmul(layer1,w2)+b2)
		layer3 = activation(tf.matmul(layer2,w3)+b3)
		layer4 = activation(tf.matmul(layer2,w4)+b4)

		out = tf.nn.sigmoid(tf.matmul(layer4, w5) + b5)
	return out

def model_train(cells,activation):
	with tf.device(DEVICE):
		layer1 = activation(tf.matmul(cells,w1)+b1)
		layer1_drop = tf.nn.dropout(layer1,0.5)
		layer2 = activation(tf.matmul(layer1_drop,w2)+b2)
		layer2_drop = tf.nn.dropout(layer2,0.5)
		layer3 = activation(tf.matmul(layer2_drop,w3)+b3)
		layer3_drop = tf.nn.dropout(layer3,0.5)
		layer4 = activation(tf.matmul(layer3_drop,w4)+b4)
		layer4_drop = tf.nn.dropout(layer4,0.5)

		out = tf.nn.sigmoid(tf.matmul(layer4_drop, w5) + b5)
	return out



# valid_boards = tf.constant(valid_boards)
# valid_labels = tf.constant(valid_labels)

# define weights, biases of each layer
w1 = tf.Variable(tf.truncated_normal(
	[board_size**2,num_hidden1], stddev=np.sqrt(2)/np.sqrt(board_size**2)))
b1 = tf.Variable(tf.zeros([num_hidden1]))

w2 = tf.Variable(tf.truncated_normal(
	[num_hidden1,num_hidden2], stddev=np.sqrt(2)/np.sqrt(num_hidden1)))
b2 = tf.Variable(tf.zeros([num_hidden2]))

w3 = tf.Variable(tf.truncated_normal(
	[num_hidden2,num_hidden3], stddev=np.sqrt(2)/np.sqrt(num_hidden2)))
b3 = tf.Variable(tf.zeros([num_hidden3]))

w4 = tf.Variable(tf.truncated_normal(
	[num_hidden3,num_hidden4], stddev=np.sqrt(2)/np.sqrt(num_hidden3)))
b4 = tf.Variable(tf.zeros([num_hidden4]))

w5 = tf.Variable(tf.truncated_normal(
	[num_hidden4,num_labels], stddev=np.sqrt(2)/np.sqrt(num_hidden4)))
b5 = tf.Variable(tf.zeros([num_labels]))

regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5)

def selu(x):

	alpha = 1.6732632423543772848170429916717
	scale = 1.0507009873554804934193349852946
	return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


logits = model_train(board_batch,selu)
# loss1 = tf.nn.softmax_cross_entropy_with_logits(
# 	labels=board_result_batch[1],
# 	logits=logits)


def manual_loss(p,q):
	xe = -tf.multiply(p, tf.log(q)) - tf.multiply((1-p), tf.log(1-q))
	return xe

#loss1 = tf.reduce_mean(tf.nn.l2_loss(logits - result_batch))
loss1 = manual_loss(result_batch,logits)
l2_l = tf.reduce_mean(tf.pow(logits - result_batch,2))

loss = tf.reduce_mean(loss1)#tf.reduce_mean(loss1)#+L2_penalty*regularizers)

if valid_boards is not None:
	valid_logits = model(valid_boards)
	valid_loss = tf.reduce_mean(manual_loss(valid_labels,valid_logits))

#optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)			
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(l2_l)
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
	l2_loss_list = []
	r_list = []

	try:
#		for n in range(N):
#			if coord.should_stop():
#				break
		while not coord.should_stop():
			
			# b,r = sess.run([board,result])
			# print b,r


			_ ,l,r,l2 = sess.run([optimizer,loss,result_batch,l2_l])
			
			loss_list.append(l)
			r_list.append(r)
			l2_loss_list.append(l2)

			if n % print_freq == 0:

				summary = sess.run(merged)
				summary_writer.add_summary(summary,n)
				l_mean = np.mean(loss_list)
				print n, l_mean, time.time(), np.mean(l2_loss_list)
				
				
				loss_list = []
				r_list = []
				l2_loss_list = []

			n += 1


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





