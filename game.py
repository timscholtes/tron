## game - contains all methods for the game

import numpy as np
import random
import itertools
from pprint import pprint

LEFT  = np.array([-1,0])
RIGHT = np.array([1,0])
UP    = np.array([0,1])
DOWN  = np.array([0,-1])


class game:
	
	def __init__(self):
		self.size = 51

	def is_legal(self,board,move,pid):
		player_loc = board['player_loc'][pid]
		new_loc = tuple(
			np.minimum(
				np.maximum(
					player_loc + move,np.array([0,0])),
					np.array([self.size-1,self.size-1])
					))

		return board['cells'][new_loc] == 2


	def legal_moves(self,board,pid):
		return [move for move in (RIGHT,LEFT,UP,DOWN) if self.is_legal(board,move,pid)]


	def is_terminal(self,board,pid):
		return len(self.legal_moves(board,pid)) == 0


	def deepish_copy(self,board):
		'''
		much, much faster than deepcopy, for a dict of the simple python types.
		need to be able to deepcopy so as to explore alternate boards.
		'''
		out = dict().fromkeys(board)
		for k,v in board.iteritems():
			# try:
			#     out[k] = v.copy()   # dicts, sets
			# except AttributeError:
			try:
				out[k] = v[:]   # lists, tuples, strings, unicode
			except TypeError:
				out[k] = v      # ints	 
		return out


	def make_move(self,board,moves):

		board_copy = self.deepish_copy(board)
		new_locs = (moves[0] + board['player_loc'][0],moves[1] + board['player_loc'][1])

		board_copy['cells'][tuple(new_locs[0])] = 0
		board_copy['cells'][tuple(new_locs[1])] = 1

		board_copy['player_loc'] = new_locs

		return board_copy

	def successors(self,board,pid):
		return [(move,self.make_move(board,move,pid)) for move in self.legal_moves(board,pid)]

	def print_board(self,board):
		opt = np.get_printoptions()
		np.set_printoptions(threshold='nan')
		pprint(board)
		np.set_printoptions(**opt)

	def play_game(self,record=False,verbose=False,*players,**kwargs):
		board = {'cells': np.full((self.size,self.size),2),
		'player_loc': (np.array([23,25]),np.array([27,25]))
		}
		board['cells'][(24,25),(23,25)] = 0
		board['cells'][(26,25),(27,25)] = 1

		while True:
			moves = [player.get_move(self,board,pid) for player,pid in zip(players,(0,1))]
			
			new_locs = (moves[0] + board['player_loc'][0],moves[1] + board['player_loc'][1])
			if (new_locs[0] == new_locs[1]).all():
				return 0
			else:
				board = self.make_move(board,moves)

			if verbose:
				self.print_board(board)

			for pid in (0,1):
				out = self.is_terminal(board,pid)
				if out:
					return 1-pid



# quick random bot:

class randomBot:

	def __init__(self):
		pass

	def get_move(self,game,board,pid):
		lmoves = game.legal_moves(board,pid)
		rm = random.randint(0,len(lmoves)-1)
		return lmoves[rm]



if __name__ == '__main__':
	g = game()
	bot1 =randomBot()
	bot2 = randomBot()
	outcome = g.play_game(False,True,bot1,bot2)
	print outcome

