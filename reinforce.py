# perform the reinforcement learning here
import random
import copy
import time
import itertools
import numpy as np

class randomBot:

	def __init__(self):
		pass

	def get_move(self,game,board,pid):
		lmoves = game.legal_moves(board,pid)
		rm = random.randint(0,len(lmoves)-1)
		return lmoves[rm]

class Reinforce:

	def __init__(self,game,policy=None,policy_history_ids=None):
		self.batch_size = 128
		if policy_history_ids is not None:
			# initialise with some random networks
			self.policies = [randomBot() for i in xrange(5)]
			for pid in policy_history_ids:
				self.policies.extend([policyNetwork.policyNetwork(game.size,pid)])
			print 'read in ', len(self.policies), 'policies'
		else:
			self.policies = [randomBot() for i in xrange(5)]
			self.policies.extend([policy])
		self.game = game
		self.opponent_update_freq = 1000


	def batch_tournament(self):
		result_store=[]
		move_store = []
		board_store = []
		nmoves_per_game = []
		while len(move_store) < self.batch_size:
			rm = random.randint(0, len(self.policies)-1)
			opponent = self.policies[rm]
			result,moves,boards = self.game.play_game(True,False,self.policies[-1],opponent)
			if result != -1:
				nmoves_per_game.append(len(moves))
				result_store.append(result)
				move_store.extend(moves)
				board_store.extend(boards)
		return result_store[:self.batch_size],move_store[:self.batch_size],board_store[:self.batch_size],nmoves_per_game

	def update_bot(self,train_boards,train_labels,train_target):

		self.policies[-1].update_pars(train_boards,train_labels,train_target)
		pass

	def prep_train_data(self,results,moves,boards,nmoves_per_game):
		
		results = (np.repeat(results,nmoves_per_game)*(-2)+1)[:self.batch_size]
		#results = np.reshape(results,(self.batch_size,1))

		# reverse boards in up-down and left right directions
		boards = [b['cells'] for b in boards]
		boards_lr = [np.fliplr(b) for b in boards]
		boards_ud = [np.flipud(b) for b in boards]
		boards_both = [np.flipud(np.fliplr(b)) for b in boards]
		boards = [boards,boards_lr,boards_ud,boards_both]
		boards = np.reshape(boards,(self.batch_size*4,self.game.size,self.game.size,1))


		flipped_move_lkup_LR = [1,0,2,3]
		flipped_move_lkup_UD = [0,1,3,2]
		flipped_move_lkup_both = [1,0,3,2]
		moves = [moves,[flipped_move_lkup_LR[m] for m in moves],
			[flipped_move_lkup_UD[m] for m in moves],
			[flipped_move_lkup_both[m] for m in moves]]
		#moves = np.reshape(moves,(self.batch_size*4))
		moves = [item for sublist in moves for item in sublist]

		results = np.transpose(np.tile(results,(1,4)))
		

		# results = (np.repeat(results,nmoves_per_game)*(-2)+1)[:self.batch_size]
		# results = np.reshape(results,(self.batch_size,1))
		# boards = np.reshape(boards,(self.batch_size,self.game.size,self.game.size,1))
		return boards, moves,results


	def train_generation(self,num_generations):
		report_res = []
		print 'N: Time:		WinRate:'
		for n in xrange(num_generations):
			results,moves,boards,nmoves_per_game = self.batch_tournament()

			# prep the data
			batch_boards,batch_labels,batch_targets = self.prep_train_data(results,moves,boards,nmoves_per_game)
			self.update_bot(batch_boards,batch_labels,batch_targets)
			report_res.extend(results)
			if n % 1000 == 0:
				print n, time.strftime("%Y-%m-%d %H:%M:%S"), np.mean(report_res)
				report_res = []

			if n < 1000 and n % 100 == 0 and n != 0:
				self.policies[-1].save_session(n)
				self.policies.append(policyNetwork.policyNetwork(self.game.size,n))
			
			if n % self.opponent_update_freq == 0 and n!= 0:
				self.policies[-1].save_session(n)
				self.policies.append(policyNetwork.policyNetwork(self.game.size,n))

			if len(self.policies) > 20:
				del self.policies[random.randint(0,len(self.policies)-3)]


if __name__ == '__main__':

	import game
	import policyNetwork
	import numpy as np



	g = game.game()


	LEFT  = np.array([0,-1])
	RIGHT = np.array([0,1])
	UP    = np.array([1,0])
	DOWN  = np.array([-1,0])

	p = policyNetwork.policyNetwork(g.size)
	# p1 = policyNetwork.policyNetwork(g.size,100)
	# p2 = policyNetwork.policyNetwork(g.size,1000)
	
	# seq = range(2,17)

	# for i in seq:
	# 	print i
	# 	p2 = policyNetwork.policyNetwork(g.size,i*100)
	# 	j = 0
	# 	res = []
	# 	while j < 20:
	# 		out = -1
	# 		while out == -1:
	# 			out = g.play_game(False,False,p1,p2)
	# 		res.append(out)
	# 		j += 1
	# 	print i, np.mean(res)


	# g.play_game(True,True,p1,p2)
	# print 'policy initialised'
	r = Reinforce(g,p)
	print 'reinforce schedule initialised'

	print 'training'
	r.train_generation(40000)
	#p2 = policyNetwork.policyNetwork(g.size,200)

	#g.play_game(True,True,p,p2)

	#p1 = policyNetwork.policyNetwork(g.size,1000)
	#p2 = policyNetwork.policyNetwork(g.size,1)

	#g.play_game(False,True,p1,p2)




