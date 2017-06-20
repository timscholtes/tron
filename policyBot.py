# this module defines the fast policy pot

class policyBot:

	def __init__(self,policy):
		self.policy = policy


	def get_move(self,game,board,pid):
		
		feed_dict = {self.policy.execute_board: np.reshape(board['cells'],[1,21,21,1])}
		move_probs = self.policy.sess.run(self.policy.execute_moves,feed_dict).tolist()[0]
	
		lmoves = game.legal_moves_index(board,pid)
		new_probs = [move_probs[i] for i in lmoves]
		new_probs = [i/sum(new_probs) for i in new_probs]
		choice = np.random.choice(lmoves,p=new_probs)
		move = (LEFT,RIGHT,UP,DOWN)[choice]
		return(move)


if __name__ == '__main__':

	import game
	import policyNetwork
	import numpy as np
	LEFT  = np.array([-1,0])
	RIGHT = np.array([1,0])
	UP    = np.array([0,1])
	DOWN  = np.array([0,-1])

	g = game.game()

	p = policyNetwork.policyNetwork(g.size)
	print p.patch_size
	board = {'cells': np.zeros((21,21)),
		'player_loc': (np.array([9,7]),np.array([9,11]))
		}
	board['cells'][(9,9),(8,7)] = (1,2)
	board['cells'][(9,9),(10,11)] = (1,3)

	pp = policyBot(p)
	print pp.get_move(g,board,0)
	