# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   11:42:41, 01-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 23:39:10, 03-Dec-2020

import numpy as np
import utils
import brain
import multiprocessing
import itertools
import time
import os

netCount = 100 # number of neural nets
gamesPer = 150; # number of oppontents each player plays each generation
fakeAgents = 50 # fake agent count
gameCount = 20 # number of games per match
## Shape:
# Input layer - memory size
# Hidden Layer
# Output Layer - move

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

if __name__ == '__main__':
	millis = str(int(round(time.time() * 1000)))
	print(f'{bcolors.HEADER}Starting generations with ID {millis}{bcolors.ENDC}')
	rawpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results', millis, 'raws');
	os.makedirs(rawpath, exist_ok=True)

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

	brainShape = [3*P*M, 5, M] # shape of neural net

# numManager.register('numeri', numeri, exposed = ['getLen', 'appendi', 'svuota', 'stampa'])  

	nets = np.array([brain.Brain.random(brainShape) for i in range(netCount)])
	# possibleGames = np.array(list(itertools.combinations(range(netCount+fakeAgents), P)), dtype=np.dtype('int,int')) # All possible arrangements of games (subsets)
	
def poolInit(a,c,f):
	global P
	global G
	global allnets
	P=a
	G=c
	allnets=f

def runGame(g, nets):
	greal = list(filter(lambda x:x<netCount, g)); # get only the "real" players
	for x in nets[greal]:x.reset()
	scores = np.zeros((len(nets),2))
	memories = np.zeros((len(allnets),int(nets[0].shape[0]/nets[0].shape[-1])))
	rcounts = np.zeros((len(nets),2))
	for j in range(gameCount):
		results = tuple(allnets[x].result(memories[x]) for x in g)
		for k in range(len(g)): # Loop through all players to calculate results
			if g[k] in greal:
				rcounts[g[k],results[k]]+=1
				scores[g[k],1]+=1 # add total plays so average score can be calculated
				scores[g[k],0]+=G[tuple(results[k:]+results[:k])] # add score based on game gridx
				memories[g[k]] = np.append(memories[g[k],P:], np.array(results[k:]+results[:k])+1)
	greal = utils.f7(greal)
	return (greal, scores, rcounts)

if __name__ == '__main__':
	def runGames():
		start = time.time();
		# reset all scores

		allnets = np.append(nets, [brain.Bot(M) for x in range(fakeAgents)]) # Create unintellectual agents

		games = [] # list of processing threads

		# NOTE: number of games per generation will be netCount choose P (can easily become huge if P>2)
		# Loop through players and choose 2 opponents for each (Note: players can play the same brains twice, including self)
		pool = multiprocessing.Pool(None, poolInit, (P,G,allnets));
		res = pool.starmap(runGame, [(np.append([i], np.random.choice(range(len(allnets)), P-1)), nets) for ii in range(gamesPer) for i in range(netCount)])
		pool.close();
		for i in res:
			for j in range(len(i[0])):
				nets[i[0][j]].updateScores(i[1][i[0][j]], i[2][i[0][j]])
		print(f'{bcolors.BOLD} Round took {time.time()-start} seconds.{bcolors.ENDC}')
	# kill half the neural nets and have the remaining half reproduce (asexual)
	def reproduce():
		scores = np.vectorize(lambda x:x.score/x.plays)(nets);
		scores = (scores+min(scores))**3 # get scores (cubed to increase weight of higher scores)
		survivors = np.random.choice(nets, netCount//2, replace=False, p=scores/sum(scores)); # chosen survivors!
		return np.append(survivors, np.vectorize(lambda x:x.reproduce())(survivors))

	def testCase(n, memory):
		c = n.calculate(memory);
		print(f"{bcolors.WARNING}Net would run {np.argmax(c[-1])+1} given {memory}; probabilities {list(c[-1])}{bcolors.ENDC}")
		print(c)

	for i in range(5000):
		# print(f"Running generation {i}...")
		runGames();
		np.save(os.path.join(rawpath, f'gen_{i}.npy'), nets)
		print(f"\n{bcolors.HEADER}Best brain for generation {i}{bcolors.ENDC}")
		n = nets[np.argmax(np.vectorize(lambda x:x.score/x.plays)(nets))]
		print(f'{bcolors.OKGREEN}Average score per game: {n.score/n.plays}{bcolors.ENDC}')
		print(f'Number of each choice: {bcolors.FAIL}{n.rcount}{bcolors.ENDC}')

		cases = [
			[0,0,0,0,0,0],
			[0,0,0,0,1,1],
			[0,0,0,0,2,1],
			[0,0,0,0,1,2],
			[0,0,1,1,1,1],
			[0,0,1,1,2,1],
			[0,0,1,1,1,2],
			[1,1,1,1,1,1],
			[1,1,1,1,1,2],
			[1,1,1,2,1,1],
			[1,2,1,2,1,1],
			[1,2,1,2,1,2],
			[2,1,2,1,2,1],
			[2,2,2,2,2,1],
			[2,2,2,1,2,1],
			[1,1,1,2,1,2],
			[1,1,1,2,2,2],
			[2,2,2,2,2,2]
		]

		for m in cases:
			testCase(n, m)

		nets = reproduce();