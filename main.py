# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   11:42:41, 01-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 18:00:49, 05-Dec-2020

import numpy as np
import utils
from utils import bcolors
import brain
import multiprocessing
import itertools
import time
import os
import shutil

netCount = 100 # number of neural nets
gamesPer = 50; # number of oppontents each player plays each generation
fakeAgents = 30 # fake agent count
gameCount = 50 # number of games per match
generations = 5000
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
		print(results)
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

	print(f'{bcolors.HEADER}Finished generations with ID {millis}{bcolors.ENDC}')
	os.rename(fpath, fpath+'-completed')
