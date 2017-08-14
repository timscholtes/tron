# perform the reinforcement learning here
import random
import copy
import time
import itertools
import numpy as np
import pickle
from game import game
import tensorflow as tf
import multiprocessing as mp

class randomBot:

	def __init__(self):
		pass

	def get_move(self,game,board,pid):
		lmoves = game.legal_moves(board,pid)
		rm = random.randint(0,len(lmoves)-1)
		return lmoves[rm]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# class MonteCarloExplore(game):
# 	def __init__(self,N,K, p1,p2):
# 		size = 21
# 		N = N
# 		K = K
# 		p1 = p1
# 		p2 = p2


g = game()

LEFT  = np.array([0,-1])
RIGHT = np.array([0,1])
UP    = np.array([1,0])
DOWN  = np.array([-1,0])

p1 = randomBot()
p2 = randomBot()

N = 1000
K = 1000

def start_board_generator():
	boards = []
	for n in range(N):
		game_boards = g.play_game_record_boards(p1,p2)
		boards.append(g.deepish_copy(random.choice(game_boards)))
	return boards

def rollout_player(start_boards):
	results = []
	for n,b in enumerate(start_boards):
		mini_res = []
		for k in range(K):
			res = g.play_game_from_board(b,p1,p2)
			if res != -1:
				mini_res.append(res)
		
		if len(mini_res) == 0:
			results.append(0.5)
		else:
			results.append(np.mean(mini_res))
	return results


def rollout_single_player(start_board):

	mini_res = []
	for k in range(K):
		res = g.play_game_from_board(start_board,p1,p2)
		if res != -1:
			mini_res.append(res)
		
	if len(mini_res) == 0:
		return 0.5
	else:
		return np.mean(mini_res)


def generate_board_results():
	boards = start_board_generator()
	results = rollout_player(boards)
	return np.stack([b['cells'] for b in boards],axis=0), results

def generate_single_board_res():

	b = start_board_generator()[0]
	res = rollout_single_player(b)
	return b['cells'],res

def outer_write_func(num_datasets,runs_per_dataset):
	for nd in range(num_datasets):
		inner_write_func(nd,runs_per_dataset)


def inner_write_func(nd):
	print 'starting ',nd
	writer = tf.python_io.TFRecordWriter('data/board_outcome_'+str(nd)+'.tfrecords')

	for rd in range(10):
		print 'generating data: ',rd, nd
		boards,results = generate_board_results()
		print 'writing data: ',rd, nd
		for b, r in zip(boards,results):
			r = np.float64(r)
			b2 = b.tolist()
			b3 = [x for y in b2 for x in y]
			example = tf.train.Example(features=tf.train.Features(
				feature = {
				'boards': _float_feature(b3),
				'results': _float_feature([r])
				}))
			writer.write(example.SerializeToString())
	writer.close()

def write_func_mp(nc,num_datasets,runs_per_dataset):

	pool = mp.Pool(processes=nc)

	for nd in range(num_datasets):
		pool.apply_async(inner_write_func,[nd])
		#pool.map(inner_write_func,range(nd))

	pool.close()
	pool.join()




if __name__ == '__main__':


	write_func_mp(32,32,50)


# 	g = game()

# 	LEFT  = np.array([0,-1])
# 	RIGHT = np.array([0,1])
# 	UP    = np.array([1,0])
# 	DOWN  = np.array([-1,0])

# 	p1 = randomBot()
# 	p2 = randomBot()

# 	MC = MonteCarloExplore(100,20,p1,p2)
# 	print 'starting MP:'
# 	#
# 	MC.write_func_mp(4,10,10)
	#MC.outer_write_func(10,10)
	


	# boards,results = MC.generate_board_results()

	# f = open('data/valid_boards.pickle', 'wb')
	# save = {
	# 'boards': boards,
	# 'results': results
	# }
	# pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	# f.close()

