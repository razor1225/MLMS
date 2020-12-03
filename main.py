# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   11:42:41, 01-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 09:24:05, 03-Dec-2020

import numpy as np
import utils
import brain
import itertools

# read game matrices
with open('dataset.txt', 'r') as f:
	tokens = list(map(int, f.read().split()));
	P = tokens.pop(0); # number of concurrent players
	M = tokens.pop(0); # number of moves per player
	G = np.zeros((P,)*M) # game payout matrix
	for i in range(P**M):
		n = utils.int2base(i, P)
		n = n+[0]*(M-len(n))
		G[tuple(n)] = tokens.pop(0)

netCount = 50 # number of neural nets
gameCount = 25 # number of games per match
brainShape = [1*P, M] # shape of neural net
## Shape:
# Input layer - memory size
# Hidden Layer
# Output Layer - move

nets = np.array([brain.Brain.random(brainShape) for i in range(netCount)])
possibleGames = list(map(list, itertools.combinations(range(netCount), P))) # All possible arrangements of games (subsets)

def runGames():
	# reset all scores
	np.vectorize(lambda x:x.resetScore())(nets)

	# NOTE: number of games per generation will be netCount choose P (can easily become huge if P>2)
	for i in possibleGames:
		# Select neural nets for game
		np.vectorize(lambda x:x.resetMemory())(nets[i])
		for j in range(gameCount):
			results = tuple(np.vectorize(lambda x:x.result())(nets[i]))
			# print(i, results)
			for k in range(P): # Loop through all players to calculate results
				nets[i[k]].score+=G[tuple(results[k:]+results[:k])] # add score based on game grid
				nets[i[k]].memory = np.append(nets[i[k]].memory[P:], results[k:]+results[:k])
	# print([x.score for x in nets])

# kill half the neural nets and have the remaining half reproduce (asexual)
def reproduce():
	scores = np.vectorize(lambda x:x.score**3)(nets); # get scores (cubed to increase weight of higher scores)
	scores = scores/sum(scores)
	survivors = np.random.choice(nets, netCount//2, replace=False, p=scores); # chosen survivors!
	return np.append(survivors, np.vectorize(lambda x:x.reproduce())(survivors))

for i in range(1000):
	# print(f"Running generation {i}...")
	runGames();
	nets = reproduce();
	np.save(f'results/{i}.npy', nets)
	print(f"\nBest brain for generation {i}:")
	n = nets[np.argmax(np.vectorize(lambda x:x.score)(nets))]
	print(n.weights)
	print(n.biases)
	print(n.score)
	print(n.rcount)
	n.memory = [-1,-1];
	print(f"First round, net would run {n.result()}")
	print(n.calculate())
	n.memory = [0,0];
	print(f"Given 5 00, net would run {n.result()}")
	print(n.calculate())
	n.memory = [0,1];
	print(f"Given 5 01, net would run {n.result()}")
	print(n.calculate())
	n.memory = [1,0];
	print(f"Given 5 10, net would run {n.result()}")
	print(n.calculate())
	n.memory = [1,1];
	print(f"Given 5 11, net would run {n.result()}")
	print(n.calculate())