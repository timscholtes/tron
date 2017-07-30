# perform the reinforcement learning here
import random
import copy
import time
import itertools
import numpy as np
import pickle
from game import game

class randomBot:

	def __init__(self):
		pass

	def get_move(self,game,board,pid):
		lmoves = game.legal_moves(board,pid)
		rm = random.randint(0,len(lmoves)-1)
		return lmoves[rm]

class MonteCarloExplore(game):
	def __init__(self,N,K, p1,p2):
		self.size = 21
		self.N = N
		self.K = K
		self.p1 = p1
		self.p2 = p2

	def start_board_generator(self):
		boards = []
		for n in range(self.N):
			print n
			game_boards = self.play_game_record_boards(self.p1,self.p2)
			boards.append(self.deepish_copy(random.choice(game_boards)))
		return boards

	def rollout_player(self,start_boards):
		results = []
		for n,b in enumerate(start_boards):
			print n
			mini_res = []
			for k in range(self.K):
				res = self.play_game_from_board(b,self.p1,self.p2)
				if res != -1:
					mini_res.append(res)
			
			if len(mini_res) == 0:
				results.append(0.5)
			else:
				results.append(np.mean(mini_res))
		return results

	def generate_board_results(self):
		boards = self.start_board_generator()
		results = self.rollout_player(boards)
		return [b['cells'] for b in boards], results







if __name__ == '__main__':


	g = game()


	LEFT  = np.array([0,-1])
	RIGHT = np.array([0,1])
	UP    = np.array([1,0])
	DOWN  = np.array([-1,0])

	p1 = randomBot()
	p2 = randomBot()

	MC = MonteCarloExplore(128*100,10,p1,p2)


	boards,results = MC.generate_board_results()

	f = open('data/train_boards.pickle', 'wb')
	save = {
	'boards': boards,
	'results': results
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()

