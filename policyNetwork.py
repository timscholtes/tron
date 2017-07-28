import tensorflow as tf
import numpy as np
import os

LEFT  = np.array([0,-1])
RIGHT = np.array([0,1])
UP    = np.array([1,0])
DOWN  = np.array([-1,0])

class policyNetwork:

	def __init__(self,board_size,restore_id=None):
		self.patch_sizes= [8,4,4]
		self.depths = [32,64,64]
		self.strides = [4,2,1]
		self.num_hidden1 = 512
		self.num_hidden2 = 256
		self.num_labels = 4
		self.board_size = board_size
		self.batch_size = 512
		self.learn_rate = 0.01
		self.L2_penalty = 0.01
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

			action_onehot = tf.one_hot(self.train_labels,4)
			action_prob = logits * action_onehot
			#action_prob = tf.gather(logits,self.train_labels)
			loss = tf.reduce_mean( - self.train_targets * tf.log(action_prob + 1e-13))
			loss = tf.reduce_mean(loss+self.L2_penalty*regularizers)

			self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(loss)

			self.execute_moves = tf.nn.softmax(self.model(self.execute_board))
			

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
		return tf.matmul(hidden2, self.w6) + self.b6

	def model_train(self,cells):
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

		return tf.matmul(hidden_dropped2, self.w6) + self.b6



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

	def softmax(self,x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def get_move(self,game,board,pid):
		
		# to make the board look the same, swap the value of the location entry
		if pid == 1:
			p1 = np.where(board['cells'] == -1)
			p2 = np.where(board['cells'] == -1)
			board['cells'][p1] = 1
			board['cells'][p2] = -1

		feed_dict = {self.execute_board: np.reshape(board['cells'],[1,21,21,1])}
		move_probs = self.sess.run(self.execute_moves,feed_dict).tolist()[0]
		# need to swap the value back
		if pid == 1:
			board['cells'][p1] = -1
			board['cells'][p2] = 1
	
		lmoves = game.legal_moves_index(board,pid)
		new_probs = [move_probs[i] for i in lmoves]
		new_probs = self.softmax(new_probs)
		choice = np.random.choice(lmoves,p=new_probs)
		move = (LEFT,RIGHT,UP,DOWN)[choice]
		# if sum(new_probs) == 0:
		# 	print '0 divisor issue: ', new_probs, move_probs
		# 	choice = np.random.choice(lmoves)
		# 	move = (LEFT,RIGHT,UP,DOWN)[choice]
		# else:
		# 	new_probs = [i/sum(new_probs) for i in new_probs]
		# 	choice = np.random.choice(lmoves,p=new_probs)
		# 	move = (LEFT,RIGHT,UP,DOWN)[choice]
		return(move)




if __name__ == '__main__':

	import game
	import numpy as np
	LEFT  = np.array([0,-1])
	RIGHT = np.array([0,1])
	UP    = np.array([1,0])
	DOWN  = np.array([-1,0])

	g = game.game()

	p1 = policyNetwork(g.size,100)
	p2 = policyNetwork(g.size,12000)
	# print p.patch_size
	# board = {'cells': np.zeros((21,21)),
	# 	'player_loc': (np.array([9,7]),np.array([9,11]))
	# 	}
	# board['cells'][(9,9),(8,7)] = (1,2)
	# board['cells'][(9,9),(10,11)] = (1,3)

	
	# move =  p.get_move(g,board,0)
	# print move

	# p.save_session(1)
	#	p.manual_restore(100)
	res = g.play_game(False,True,p1,p2)
	print res


	
