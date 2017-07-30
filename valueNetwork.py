import tensorflow as tf
import numpy as np
import os
import pickle

LEFT  = np.array([0,-1])
RIGHT = np.array([0,1])
UP    = np.array([1,0])
DOWN  = np.array([-1,0])
DEVICE = '/cpu:0'

class valueNetwork:

	def __init__(self,board_size,restore_id=None,
		valid_boards=None,valid_labels=None):
		self.patch_sizes= [8,4,4]
		self.depths = [32,64,64]
		self.strides = [4,2,1]
		self.num_hidden1 = 512
		self.num_hidden2 = 256
		self.num_labels = 1
		self.board_size = board_size
		self.batch_size = 512
		self.learn_rate = 0.01
		self.L2_penalty = 0.01
		self.graph = tf.Graph()

		with self.graph.as_default():

			self.batch_boards = tf.placeholder(
				tf.float32,shape=(self.batch_size,self.board_size,self.board_size,1))
			self.batch_labels = tf.placeholder(tf.int32)
			if valid_boards is not None:
				valid_boards = tf.constant(valid_boards)
				valid_labels = tf.constant(valid_labels)

			# define weights, biases of each layer
			self.w1 = tf.Variable(tf.truncated_normal(
				[self.patch_sizes[0], self.patch_sizes[0], 1, self.depths[0]], stddev=0.1))
			self.b1 = tf.Variable(tf.zeros([self.depths[0]]))

			self.w2 = tf.Variable(tf.truncated_normal(
				[self.patch_sizes[1], self.patch_sizes[1], self.depths[0], self.depths[1]], stddev=0.1))
			self.b2 = tf.Variable(tf.constant(1.0,shape=[self.depths[1]]))

			self.w3 = tf.Variable(tf.truncated_normal(
				[self.patch_sizes[2], self.patch_sizes[2], self.depths[1], self.depths[2]], stddev=0.1))
			self.b3 = tf.Variable(tf.constant(1.0,shape=[self.depths[2]]))

			self.w4 = tf.Variable(tf.truncated_normal(
				# [self.board_size // 4 * self.board_size // 4 * self.depth, self.num_hidden], stddev=0.1))
				[6*6*16, self.num_hidden1], stddev=0.1))
			self.b4 = tf.Variable(tf.constant(1.0,shape=[self.num_hidden1]))

			self.w5 = tf.Variable(tf.truncated_normal(
				[self.num_hidden1, self.num_hidden2], stddev=0.1))
			self.b5 = tf.Variable(tf.constant(1.0,shape=[self.num_hidden2]))

			self.w6 = tf.Variable(tf.truncated_normal(
				[self.num_hidden2, self.num_labels], stddev=0.1))
			self.b6 = tf.Variable(tf.constant(1.0,shape=[self.num_labels]))

			regularizers = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2) + \
				 tf.nn.l2_loss(self.w3) + tf.nn.l2_loss(self.w4) + \
				 tf.nn.l2_loss(self.w5) +tf.nn.l2_loss(self.w6)


			logits = self.model_train(self.train_boards)
			# loss = tf.reduce_mean(
			# 	tf.nn.softmax_cross_entropy_with_logits(labels=train_labels,logits=logits))

			# action_onehot = tf.one_hot(self.train_labels,4)
			# action_prob = logits * action_onehot
			# #action_prob = tf.gather(logits,self.train_labels)
			# loss = tf.reduce_mean( - self.train_targets * tf.log(action_prob + 1e-13))

			self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				labels=self.train_labels,logits=logits))

			loss = tf.reduce_mean(self.loss1+self.L2_penalty*regularizers)

			if valid_boards is not None:
				valid_logits = self.model(valid_boards)
				self.valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
					labels=valid_labels,logits=valid_logits))

			self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(loss)			

			self.sess = tf.Session(graph=self.graph)
			
			self.saver = tf.train.Saver()
			if restore_id is not None:
				#print 'restoring from: ',"/models/model"+str(restore_id)+".ckpt"
				self.saver.restore(self.sess,"./models/model"+str(restore_id)+".ckpt")
			else:
				print 'initializing'
				init = tf.global_variables_initializer()
				self.sess.run(init)
			

	def model(self,cells):
		with tf.device(DEVICE):
			conv = tf.nn.conv2d(cells, self.w1, [1, 4, 4, 1], padding='SAME')
			hidden = tf.nn.relu(conv + self.b1)
			conv = tf.nn.conv2d(hidden, self.w2, [1, 2, 2, 1], padding='SAME')
			hidden = tf.nn.relu(conv + self.b2)
			conv = tf.nn.conv2d(hidden, self.w3, [1, 1, 1, 1], padding='SAME')
			hidden = tf.nn.relu(conv + self.b3)

			shape = hidden.get_shape().as_list()
			reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
			hidden1 = tf.nn.relu(tf.matmul(reshape, self.w4) + self.b4)
			hidden2 = tf.nn.relu(tf.matmul(hidden1, self.w5) + self.b5)
			out =  tf.nn.softmax(tf.matmul(hidden2, self.w6) + self.b6)
		return out

	def model_train(self,cells):
		with tf.device(DEVICE):
			conv = tf.nn.conv2d(cells, self.w1, [1, 4, 4, 1], padding='SAME')
			hidden = tf.nn.relu(conv + self.b1)
			conv = tf.nn.conv2d(hidden, self.w2, [1, 2, 2, 1], padding='SAME')
			hidden = tf.nn.relu(conv + self.b2)
			conv = tf.nn.conv2d(hidden, self.w3, [1, 1, 1, 1], padding='SAME')
			hidden = tf.nn.relu(conv + self.b3)

			shape = hidden.get_shape().as_list()
			reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
			hidden1 = tf.nn.relu(tf.matmul(reshape, self.w4) + self.b4)
			hidden_dropped1 = tf.nn.dropout(hidden1,0.5)
			hidden2 = tf.nn.relu(tf.matmul(hidden_dropped1, self.w5) + self.b5)
			hidden_dropped2 = tf.nn.dropout(hidden2,0.5)

			out =  tf.nn.softmax(tf.matmul(hidden_dropped2, self.w6) + self.b6)
		return out


	def update_pars(self,train_boards,train_labels,N, batch_size):
		print 'Step:','Train_loss','Valid_loss'
		for n in N:
			offset = (n * batch_size) % (self.train_labels.shape[0] - batch_size)
			# Generate a minibatch.
			batch_boards = self.train_boards[offset:(offset + batch_size), :]
			batch_labels = self.train_labels[offset:(offset + batch_size), :]
			feed_dict = {self.batch_boards: batch_boards,
			self.batch_labels: batch_labels}
			self.sess.run(self.optimizer,feed_dict)
			if n % 100 == 0:
				train_loss = self.loss1.eval()
				valid_loss = self.valid_loss.eval()
				print n, train_loss, valid_loss
		pass


	def save_session(self,id):
		save_path = self.saver.save(self.sess,"./models/model"+str(id)+".ckpt")
		print("Model saved in file: %s" % save_path)
		pass

	def manual_restore(self,restore_id):
		print 'restoring from: ',"./models/model"+str(restore_id)+".ckpt"
		self.saver.restore(self.sess,"./models/model"+str(restore_id)+".ckpt")
		pass




if __name__ == '__main__':

	import game
	import numpy as np
	LEFT  = np.array([0,-1])
	RIGHT = np.array([0,1])
	UP    = np.array([1,0])
	DOWN  = np.array([-1,0])

	with open('data/train_boards.pickle', 'rb') as f:
		data = pickle.load(f)
		train_boards = data['boards']
		train_results = data['results']
		del data

	with open('data/valid_boards.pickle', 'rb') as f:
		data = pickle.load(f)
		valid_boards = data['boards']
		valid_results = data['results']
		del data

	VN = valueNetwork(21,None,valid_boards,valid_results)


