# perform the reinforcement learning here
import random
import copy

class Reinforce:

	def __init__(self,policy,game):
		self.batch_size = 12
		self.policies = [policy]
		self.game = game
		self.num_generations = 100
		self.opponent_update_freq = 100


	def batch_tournament(self):
		rm = random.randint(0, len(self.policies)-1)
		opponent = self.policies[rm]
		result_store=[]
		move_store = []
		board_store = []
		while len(result_store) < self.batch_size:
			result,moves,boards = self.game.play_game(True,False,self.policies[-1],opponent)
			if result != -1:
				result_store.append(result)
				move_store.append(moves)
				board_store.append(boards)
		return result_store[:self.batch_size],move_store[:self.batch_size],board_store[:self.batch_size]

	def update_bot(self,train_boards,train_labels):
		self.policies[-1].update_pars(train_boards,train_labels)
		pass

	def prep_results(self,results,moves,boards):
		ngames = len(results)
		pass
		#for n in ngames:




	def train_generation(self):
		for n in xrange(self.num_generations):
			results,moves,boards = self.batch_tournament()

			# prep the data
			train_boards,train_labels = self.prep_data(results,moves,boards)
			self.update_bot(train_boards,train_labels)

			if n % 100 == 0:
				print n

			if n % self.opponent_update_freq == 0 and n != 0:
				self.policies[-1].save_session(n)
				self.policies.append(copy.deepcopy(self.policies[-1]))
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

	a,b,c = r.batch_tournament()

	print a,b








