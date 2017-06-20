# this module defines the fast policy pot

class policyBot:

	def __init__(self,model):
		self.model = model

	def get_move(self,game,board,pid):
		lmoves = game.legal_moves_index(board,pid)
		
		probs = self.model(board)[lmoves]
		choice = np.random.choice(lmoves,probs)
		move = (LEFT,RIGHT,UP,DOWN)[choice]

		return move

