import tensorflow as tf
import numpy as np
import os
LEFT  = np.array([-1,0])
RIGHT = np.array([1,0])
UP    = np.array([0,1])
DOWN  = np.array([0,-1])

class policyNetwork:

	def __init__(self,board_size,restore_id=None):
		self.patch_size=5
		self.depth = 16
		self.num_hidden = 64
		self.num_labels = 4
		self.board_size = board_size
		self.batch_size = 128
		self.learn_rate = 0.01
		self.graph = tf.Graph()

		with self.graph.as_default():

			self.train_boards = tf.placeholder(
				tf.float32,shape=(self.batch_size,self.board_size,self.board_size,1))
			self.train_labels = tf.placeholder(tf.int32)
			self.train_targets = tf.placeholder(tf.float32)
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

			logits = self.model(self.train_boards)
			# loss = tf.reduce_mean(
			# 	tf.nn.softmax_cross_entropy_with_logits(labels=train_labels,logits=logits))

			action_onehot = tf.one_hot(self.train_labels,4)
			action_prob = logits * action_onehot
			#action_prob = tf.gather(logits,self.train_labels)
			loss = tf.reduce_mean( - self.train_targets * tf.log(action_prob + 1e-13))

			self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(loss)

			self.execute_moves = tf.nn.softmax(self.model(self.execute_board))
			

			self.sess = tf.Session(graph=self.graph)
			
			self.saver = tf.train.Saver()
			if restore_id is not None:
				print 'restoring from: ',"/models/model"+str(restore_id)+".ckpt"
				self.saver.restore(self.sess,"./models/model"+str(restore_id)+".ckpt")
			else:
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

	

	def update_pars(self,batch_boards,batch_labels,batch_targets):
		feed_dict = {self.train_boards: batch_boards,
		self.train_labels: batch_labels,
		self.train_targets: batch_targets}
		self.sess.run(self.optimizer,feed_dict)
		pass

	def save_session(self,id):
		save_path = self.saver.save(self.sess,"./models/model"+str(id)+".ckpt")
		print("Model saved in file: %s" % save_path)
		pass

	def manual_restore(self,restore_id):
		print 'restoring from: ',"./models/model"+str(restore_id)+".ckpt"
		self.saver.restore(self.sess,"./models/model"+str(restore_id)+".ckpt")
		pass

	def get_move(self,game,board,pid):
		
		feed_dict = {self.execute_board: np.reshape(board['cells'],[1,21,21,1])}
		move_probs = self.sess.run(self.execute_moves,feed_dict).tolist()[0]
	
		lmoves = game.legal_moves_index(board,pid)
		new_probs = [move_probs[i] for i in lmoves]
		new_probs = [i/sum(new_probs) for i in new_probs]
		choice = np.random.choice(lmoves,p=new_probs)
		move = (LEFT,RIGHT,UP,DOWN)[choice]
		return(move)




if __name__ == '__main__':

	import game
	import numpy as np
	LEFT  = np.array([-1,0])
	RIGHT = np.array([1,0])
	UP    = np.array([0,1])
	DOWN  = np.array([0,-1])

	g = game.game()

	p = policyNetwork(g.size,1)
	print p.patch_size
	board = {'cells': np.zeros((21,21)),
		'player_loc': (np.array([9,7]),np.array([9,11]))
		}
	board['cells'][(9,9),(8,7)] = (1,2)
	board['cells'][(9,9),(10,11)] = (1,3)

	
	move =  p.get_move(g,board,0)
	print move

	p.save_session(1)
	p.manual_restore(1)

	
