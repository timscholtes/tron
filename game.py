## game - contains all methods for the game

import numpy as np


LEFT  = (-1,0)
RIGHT = (1,0)
UP    = (0,1)
DOWN  = (0,-1)


class game:
	
	def __init__(self):
		# self.size = 51
		# self.board = np.zeros((self.size,self.size))
		pass
		# self.p1Pos = (25,25)
		# self.p2Pos = (26,26)
		# self.p1Dir = self.LEFT
		# self.p2Dir = self.RIGHT
		


	def legal_moves(self,board,pid):
		player_loc = board['player_loc'][pid]
		return [move for move in (RIGHT,LEFT,UP,DOWN) if
		 board['cells'][player_loc + move] == 0]


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

	def make_move(self,board,move,pid):
		board_copy = self.deepish_copy(board)

		player_loc = board_copy['player_loc'][pid]
		board_copy['cells'][player_loc]






if __name__ == '__main__':
	g = game()
	print g.startDir

