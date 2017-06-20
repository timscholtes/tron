import tensorflow as tf
import numpy as np
LEFT  = np.array([-1,0])
RIGHT = np.array([1,0])
UP    = np.array([0,1])
DOWN  = np.array([0,-1])
class policyNetwork:

	def __init__(self,board_size):
		self.patch_size=5
		self.depth = 16
		self.num_hidden = 64
		self.num_labels = 4
		self.board_size = board_size
		self.batch_size = 128*10
		self.learn_rate = 0.01
		self.graph = tf.Graph()

		with self.graph.as_default():

			train_boards = tf.placeholder(
				tf.float32,shape=(self.batch_size,self.board_size,self.board_size,1))
			train_labels = tf.placeholder(tf.float32,shape=(self.batch_size,self.num_labels))
			self.execute_board = tf.placeholder(
				tf.float32,shape=(1,self.board_size,self.board_size,1))

			# define weights, biases of each layer
			self.w1 = tf.Variable(tf.truncated_normal(
				[self.patch_size, self.patch_size, 1, self.depth], stddev=0.1))
			self.b1 = tf.Variable(tf.zeros([self.depth]))

			self.w2 = tf.Variable(tf.truncated_normal(
				[self.patch_size, self.patch_size, self.depth, self.depth], stddev=0.1))
			self.b2 = tf.Variable(tf.constant(1.0,shape=[self.depth]))

			self.w3 = tf.Variable(tf.truncated_normal(
				# [self.board_size // 4 * self.board_size // 4 * self.depth, self.num_hidden], stddev=0.1))
				[6*6*16, self.num_hidden], stddev=0.1))
			self.b3 = tf.Variable(tf.constant(1.0,shape=[self.num_hidden]))

			self.w4 = tf.Variable(tf.truncated_normal(
				[self.num_hidden, self.num_labels], stddev=0.1))
			self.b4 = tf.Variable(tf.constant(1.0,shape=[self.num_labels]))

			logits = self.model(train_boards)
			loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=train_labels,logits=logits))
			self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(loss)

			self.execute_moves = tf.nn.softmax(self.model(self.execute_board))

			self.sess = tf.Session(graph=self.graph)
			print 'initializing'
			init = tf.global_variables_initializer()
			self.sess.run(init)

	def model(self,cells):
		conv = tf.nn.conv2d(cells, self.w1, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + self.b1)
		conv = tf.nn.conv2d(hidden, self.w2, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + self.b2)
		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, self.w3) + self.b3)
		return tf.matmul(hidden, self.w4) + self.b4

	

	def update_pars(self,train_boards,):
		feed_dict = {train_boards: train_boards,
		train_labels: train_labels}
		self.sess.run(self.optimizer)
		pass





