## game - contains all methods for the game

import numpy as np
import random
import itertools
from pprint import pprint
import copy

LEFT  = np.array([0,-1])
RIGHT = np.array([0,1])
UP    = np.array([1,0])
DOWN  = np.array([-1,0])


class game:
	
	def __init__(self):
		self.size = 21

	def is_legal(self,board,move,pid):
		player_loc = board['player_loc'][pid]
		new_loc = tuple(
			np.minimum(
				np.maximum(
					player_loc + move,np.array([0,0])),
					np.array([self.size-1,self.size-1])
					))

		return board['cells'][new_loc] == -0.5


	def legal_moves(self,board,pid):
		return [move for move in (LEFT,RIGHT,UP,DOWN) if self.is_legal(board,move,pid)]

	def legal_moves_index(self,board,pid):
		return [i for move,i in zip([LEFT,RIGHT,UP,DOWN],range(4)) if self.is_legal(board,move,pid)]

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
			if type(v) == np.ndarray:
				out[k] = 1*v
			else:
				try:
					out[k] = v[:]   # lists, tuples, strings, unicode
				except TypeError:
					out[k] = v      # ints	 
		return out


	def make_move(self,board,moves):

		board_copy = self.deepish_copy(board)
		cur_locs = board['player_loc']
		new_locs = (moves[0] + cur_locs[0],moves[1] + cur_locs[1])

		board_copy['cells'][tuple(new_locs[0])] = -1
		board_copy['cells'][tuple(new_locs[1])] = 1
		board_copy['cells'][tuple(cur_locs[0])] = 0.5
		board_copy['cells'][tuple(cur_locs[1])] = 0.5

		board_copy['cells']

		board_copy['player_loc'] = new_locs

		return board_copy

	def successors(self,board,pid):
		return [(move,self.make_move(board,move,pid)) for move in self.legal_moves(board,pid)]


	def print_board(self,board):
		print '-'*(self.size*3+2)
		for y in range(self.size):
			line=['|']
			for x in range(self.size):
				v = board['cells'][y,x]
				if v == -0.5:
					a = ' '
				elif v == 0.5:
					a = '*'
				elif v == -1:
					a = '1'
				elif v == 1:
					a = '2'
				line.append(' {} '.format(a))
			line.append('|')
			print ''.join(line)
		print '-'*(self.size*3+2)
		pass


	def play_game(self,record=False,verbose=False,*players,**kwargs):
		p1 = np.random.choice([-1,1])

		board = {'cells': np.full((self.size,self.size),-0.5),
		# 'player_loc': (np.array([9,7]),np.array([9,11]))
		'player_loc': (np.array([9,9+2*p1]),np.array([9,9-2*p1]))
		}
		

		# board['cells'][(9,9),(8,7)] = (0.5,-1)
		# board['cells'][(9,9),(10,11)] = (0.5,1)
		board['cells'][(9,9),(8,7)] = (0.5,p1)
		board['cells'][(9,9),(10,11)] = (0.5,-1*p1)

		all_moves = list()
		all_boards = list()
		if verbose:
			print board['player_loc']
			self.print_board(board)
		while True:
			moves = [player.get_move(self,board,pid) for player,pid in zip(players,(0,1))]
			if record:
				board_copy = self.deepish_copy(board)
				all_boards.append(board_copy)
				move_ind = [i for i,m in enumerate([LEFT,RIGHT,UP,DOWN]) if (m == moves[0]).all()]
				all_moves.extend(move_ind)
			new_locs = (moves[0] + board['player_loc'][0],moves[1] + board['player_loc'][1])
			if (new_locs[0] == new_locs[1]).all():
				if record:
					return -1,all_moves,all_boards
				else:
					return -1
			else:
				board = self.make_move(board,moves)

			if verbose:
				print board['player_loc']
				self.print_board(board)

			for pid in (0,1):
				out = self.is_terminal(board,pid)
				if out:
					if record:
						return 1-pid, all_moves,all_boards
					else:
						return 1-pid

	def play_game_record_boards(self,*players,**kwargs):
		p1 = np.random.choice([-1,1])

		board = {'cells': np.full((self.size,self.size),-0.5),
		# 'player_loc': (np.array([9,7]),np.array([9,11]))
		'player_loc': (np.array([9,9+2*p1]),np.array([9,9-2*p1]))
		}
		
		board['cells'][(9,9),(8,7)] = (0.5,p1)
		board['cells'][(9,9),(10,11)] = (0.5,-1*p1)

		all_boards = list()
		while True:
			moves = [player.get_move(self,board,pid) for player,pid in zip(players,(0,1))]
			board_copy = self.deepish_copy(board)
			all_boards.append(board_copy)
			new_locs = (moves[0] + board['player_loc'][0],moves[1] + board['player_loc'][1])
			if (new_locs[0] == new_locs[1]).all():
				return all_boards
			else:
				board = self.make_move(board,moves)

			for pid in (0,1):
				out = self.is_terminal(board,pid)
				if out:
					return all_boards

	def play_game_from_board(self,board,*players,**kwargs):
		while True:
			moves = [player.get_move(self,board,pid) for player,pid in zip(players,(0,1))]
			new_locs = (moves[0] + board['player_loc'][0],moves[1] + board['player_loc'][1])
			if (new_locs[0] == new_locs[1]).all():
				return -1
			else:
				board = self.make_move(board,moves)

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
