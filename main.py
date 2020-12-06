# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   11:42:41, 01-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 14:18:05, 06-Dec-2020

import numpy as np
import utils
from utils import bcolors
import brain
import multiprocessing
import itertools
import time
import os
import shutil

netCount = 10 # number of neural nets
gamesPer = 10; # number of oppontents each player plays each generation
fakeAgents = 2 # fake agent count
gameCount = 30 # number of games per match
generations = 100
## Shape:
# Input layer - memory size
# Hidden Layer
# Output Layer - move

if __name__ == '__main__':
	millis = str(int(round(time.time() * 1000)))
	print(f'{bcolors.HEADER}Starting generations with ID {millis}{bcolors.ENDC}')
	fpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results', millis)
	rawpath = os.path.join(fpath, 'raws');
	os.makedirs(rawpath, exist_ok=True)

	# read game matrices
	shutil.copy('dataset.txt', os.path.join(fpath, 'dataset.txt'));
	with open('dataset.txt', 'r') as f:
		tokens = list(map(int, f.read().split()));
		P = tokens.pop(0); # number of concurrent players
		M = tokens.pop(0); # number of moves per player
		G = np.zeros((M,)*P) # game payout matrix
		for i in range(M**P):
			n = utils.int2base(i, M)
			n = n+[0]*(P-len(n))
			G[tuple(n)] = tokens.pop(0)

	brainShape = [5*P*M, 10, M] # shape of neural net
	
	np.save(os.path.join(fpath, f'params.npy'), (netCount, gamesPer, fakeAgents, gameCount, generations, P, M, G, brainShape))
	multiprocessing.set_start_method('spawn')

# numManager.register('numeri', numeri, exposed = ['getLen', 'appendi', 'svuota', 'stampa'])  

	nets = np.array([brain.Brain.random(brainShape) for i in range(netCount)])
	# possibleGames = np.array(list(itertools.combinations(range(netCount+fakeAgents), P)), dtype=np.dtype('int,int')) # All possible arrangements of games (subsets)
	
def poolInit(a,c,d):
	global P
	global G
	global M
	P=a
	G=c
	M=d

def runGame(g, allnets):
	greal = list(filter(lambda x:x<netCount, g)); # get only the "real" players
	for x in allnets[greal]:x.reset()
	scores = np.zeros((netCount,2))
	memories = np.zeros((len(allnets),int(allnets[0].shape[0]/M)))
	rcounts = np.zeros((netCount,M))
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
	pool = multiprocessing.Pool(None, poolInit, (P,G,M));
	def runGames():
		start = time.time();
		# reset all scores

		allnets = np.append(nets, [brain.Bot(M) for x in range(fakeAgents)]) # Create unintellectual agents

		games = [] # list of processing threads

		# NOTE: number of games per generation will be netCount choose P (can easily become huge if P>2)
		# Loop through players and choose 2 opponents for each (Note: players can play the same brains twice, including self)
		res = pool.starmap(runGame, [(np.append([i], np.random.choice(range(len(allnets)), P-1)), allnets) for ii in range(gamesPer) for i in range(netCount)])
		for i in res:
			for j in range(len(i[0])):
				nets[i[0][j]].updateScores(i[1][i[0][j]], i[2][i[0][j]])
		print(f'{bcolors.BOLD}Round took {time.time()-start} seconds.{bcolors.ENDC}')
	# kill half the neural nets and have the remaining half reproduce (asexual)
	def reproduce():
		scores = np.vectorize(lambda x:x.score/x.plays)(nets);
		scores = (scores-min(scores))**3 # get scores (cubed to increase weight of higher scores)
		survivors = np.random.choice(nets, netCount//2, replace=False, p=scores/sum(scores)); # chosen survivors!
		return np.append(survivors, np.vectorize(lambda x:x.reproduce())(survivors))

	for i in range(generations):
		print(f"\n\n{bcolors.HEADER}Running generation {i}...{bcolors.ENDC}")
		runGames();
		np.save(os.path.join(rawpath, f'gen_{i}.npy'), nets)
		print(f"{bcolors.HEADER}Best brain for generation {i}{bcolors.ENDC}")
		n = nets[np.argmax(np.vectorize(lambda x:x.score/x.plays)(nets))]
		print(f'{bcolors.OKGREEN}Average score per game: {n.score/n.plays}{bcolors.ENDC}')
		print(f'Number of each choice: {bcolors.FAIL}{n.rcount}{bcolors.ENDC}')

		for m in brain.cases:
			if len(m) == brainShape[0]/M:
				brain.testCase(n, m)

		nets = reproduce();

	pool.close();
	print(f'{bcolors.HEADER}Finished generations with ID {millis}{bcolors.ENDC}')
	os.rename(fpath, fpath+'-completed')
