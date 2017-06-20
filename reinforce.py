# perform the reinforcement learning here
import random

class Reinforce:

	def __init__(self,policy,game):
		self.batch_size = 128
		self.policies = [policy]
		self.game = game


	def batch_tournament(self):
		rm = random.randint(0, len(self.policies)-1)
		opponent = self.policies[rm]
		#opponent = np.random.choice(self.policies)
		results=[]
		# first pass
		for match in xrange(self.mini_batch_size):
			results.append(
				self.game.play_game(True,False,self.policies[len(self.policies)],opponent))
		return results

	def update_bot(self,train_boards,train_labels):
		self.policies[len(self.policies)].update_pars(train_boards,train_labels)
		pass

	def train_generation

