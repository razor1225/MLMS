# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   11:42:41, 01-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 18:16:50, 03-Dec-2020

import numpy as np
import utils
import brain
import itertools

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
gamesPer = 40; # number of oppontents each player plays each generation
fakeAgents = 25 # fake agent count
gameCount = 15 # number of games per match
brainShape = [2*P*M, 5, M] # shape of neural net
## Shape:
# Input layer - memory size
# Hidden Layer
# Output Layer - move

nets = np.array([brain.Brain.random(brainShape) for i in range(netCount)])
# possibleGames = np.array(list(itertools.combinations(range(netCount+fakeAgents), P)), dtype=np.dtype('int,int')) # All possible arrangements of games (subsets)

def runGames():
	# reset all scores
	np.vectorize(lambda x:x.resetScore())(nets)

	allnets = np.append(nets, [brain.Bot(M) for x in range(fakeAgents)]) # Create unintellectual agents

	# NOTE: number of games per generation will be netCount choose P (can easily become huge if P>2)
	# Loop through players and choose 2 opponents for each (Note: players can play the same brains twice, including self)
	for i in range(netCount):
		for ii in range(gamesPer):
			# Select neural nets for game
			g = np.append([i], np.random.choice(range(len(allnets)), P-1))
			greal = list(filter(lambda x:x<netCount, g)); # get only the "real" players
			np.vectorize(lambda x:x.resetMemory())(nets[greal])
			for j in range(gameCount):
				results = tuple(np.vectorize(lambda x:x.result())(allnets[g]))
				# print(i, results)
				for k in range(len(g)): # Loop through all players to calculate results
					if g[k] in greal:
						nets[g[k]].plays+=1 # add total plays so average score can be calculated
						nets[g[k]].score+=G[tuple(results[k:]+results[:k])] # add score based on game gridx
						nets[g[k]].memory = np.append(nets[g[k]].memory[P:], np.array(results[k:]+results[:k])+1)

# kill half the neural nets and have the remaining half reproduce (asexual)
def reproduce():
	scores = np.vectorize(lambda x:x.score/x.plays)(nets);
	scores = (scores+min(scores))**3 # get scores (cubed to increase weight of higher scores)
	survivors = np.random.choice(nets, netCount//2, replace=False, p=scores/sum(scores)); # chosen survivors!
	return np.append(survivors, np.vectorize(lambda x:x.reproduce())(survivors))

def testCase(n, memory):
	n.memory = memory;
	c = n.calculate();
	print(f"{bcolors.WARNING}Net would run {np.argmax(c[-1])+1} given {memory}; probabilities {list(c[-1])}{bcolors.ENDC}")
	print(c)


runGames();

for i in range(1000):
	# print(f"Running generation {i}...")
	runGames();
	np.save(f'results/{i}.npy', nets)
	print(f"\n{bcolors.HEADER}Best brain for generation {i}{bcolors.ENDC}")
	n = nets[np.argmax(np.vectorize(lambda x:x.score/x.plays)(nets))]
	print(f'{bcolors.OKGREEN}Average score per game: {n.score/n.plays}{bcolors.ENDC}')
	print(f'Number of each choice: {bcolors.FAIL}{n.rcount}{bcolors.ENDC}')

	cases = [
		[0,0,0,0],
		[0,0,1,1],
		[0,0,2,1],
		[0,0,1,2],
		[0,0,2,2],
		[1,1,1,1],
		[1,1,1,2],
		[1,2,1,2],
		[2,1,2,1],
		[2,2,2,1],
		[1,1,2,1],
		[1,2,1,1],
		[2,2,2,2]
	]

	for m in cases:
		testCase(n, m)

	nets = reproduce();