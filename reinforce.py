# perform the reinforcement learning here
import random
import copy

class Reinforce:

	def __init__(self,policy,game):
		self.batch_size = 128
		self.policies = [policy]
		self.game = game
		self.num_generations = 101
		self.opponent_update_freq = 10


	def batch_tournament(self):
		rm = random.randint(0, len(self.policies)-1)
		opponent = self.policies[rm]
		result_store=[]
		move_store = []
		board_store = []
		nmoves_per_game = []
		while len(move_store) < self.batch_size:
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
		results = np.reshape(results,(self.batch_size,1))
		boards = np.reshape(boards,(self.batch_size,self.game.size,self.game.size,1))
		return boards, moves,results


	def train_generation(self):
		for n in xrange(self.num_generations):
			results,moves,boards,nmoves_per_game = self.batch_tournament()

			# prep the data
			batch_boards,batch_labels,batch_targets = self.prep_train_data(results,moves,boards,nmoves_per_game)
			self.update_bot(batch_boards,batch_labels,batch_targets)

			if n % 10 == 0:
				print n

			if n % self.opponent_update_freq == 0 and n != 0:
				self.policies[-1].save_session(n)
				self.policies.append(policyNetwork.policyNetwork(self.game.size,n))
			if len(self.policies) > 10:
				del self.policies[random.randint(0,len(self.policies)-3)]


if __name__ == '__main__':

	import game
	import policyNetwork
	import numpy as np



	g = game.game()

	LEFT  = np.array([-1,0])
	RIGHT = np.array([1,0])
	UP    = np.array([0,1])
	DOWN  = np.array([0,-1])

	p = policyNetwork.policyNetwork(g.size)

	r = Reinforce(p,g)

	r.train_generation()








