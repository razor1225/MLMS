# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:	 22:05:55, 02-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 14:01:05, 07-Dec-2020

import numpy as np
import utils
from utils import bcolors

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
	[2,2,2,2],
	[0,0,0,0,0,0],
	[0,0,0,0,1,1],
	[0,0,0,0,2,1],
	[0,0,0,0,1,2],
	[0,0,1,1,1,1],
	[1,1,1,1,1,1],
	[1,1,1,1,2,1],
	[1,1,1,1,1,2],
	[1,1,1,2,1,1],
	[1,2,1,2,1,1],
	[1,2,1,2,1,2],
	[2,1,2,1,2,1],
	[2,2,2,2,2,1],
	[2,2,2,1,2,1],
	[1,1,1,2,1,2],
	[1,1,1,2,2,2],
	[2,2,2,2,2,1],
	[2,2,2,2,1,2],
	[2,2,2,2,2,2],
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,1,1],
	[0,0,0,0,0,0,0,0,2,2],
	[0,0,0,0,0,0,0,0,1,2],
	[0,0,0,0,0,0,0,0,2,1],
	[1,1,1,1,1,1,1,1,1,1],
	[1,1,1,1,1,1,1,1,2,1],
	[1,1,1,1,1,1,1,1,1,2],
	[1,1,1,1,1,1,1,1,2,2],
	[1,1,1,1,1,1,2,2,1,1],
	[1,2,1,2,1,2,1,2,1,2],
	[2,1,2,1,2,1,2,1,2,1],
	[2,2,1,1,2,2,1,1,2,2],
	[1,2,1,2,1,2,1,2,2,1],
	[2,1,2,1,2,1,2,1,1,2],
	[2,2,2,2,2,2,2,2,2,1],
	[2,2,2,2,2,2,2,2,1,2],
	[2,2,2,2,2,2,2,2,2,2],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
	[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
	[1,1,2,2,1,2,1,1,2,2,1,2,2,1,2],
	[2,2,1,2,2,1,2,2,1,2,2,1,2,2,1],
	[2,1,1,2,1,1,2,1,1,2,1,1,2,1,1],
	[1,2,1,1,2,1,1,2,1,1,2,1,1,2,1],
	[1,2,2,1,2,2,1,2,2,1,2,2,1,2,2],
	[1,2,2,1,2,2,1,2,2,1,1,2,1,1,2]
]

# chance to become a random agent
random_chance = 0.01
# chance to completely change a weight
big_mutate_chance = 0.0005;
# chance to mutate a weight by value*small_mutate_prop*randn
small_mutate_chance = 0.001
small_mutate_prop = 0.01;

def mutateNode(n):
	if np.random.sample() < small_mutate_chance:
		n=n*(1+small_mutate_prop*np.random.randn());
	if np.random.sample() < big_mutate_chance:
		n=np.random.randn();
	return n;

def runGame(nets, count, M, G):
	for x in nets:x.reset()
	scores = np.zeros((len(nets),2))
	memories = np.zeros((len(nets),int(nets[0].shape[0]/M)))
	rcounts = np.zeros((len(nets),M))
	moves = np.zeros((count,len(nets)), dtype=int)
	allscores = np.zeros((count, len(nets)))
	for j in range(count):
		moves[j] = [nets[x].result(memories[x]) for x in range(len(nets))]
		for k in range(len(nets)): # Loop through all players to calculate moves
			rcounts[k,moves[j,k]]+=1
			shifted = tuple(moves[j,k:])+tuple(moves[j,:k]);
			scores[k,1]+=1 # add total plays so average score can be calculated
			allscores[j,k] = G[shifted]
			scores[k,0]+=allscores[j,k] # add score based on game gridx
			memories[k] = np.append(memories[k,len(nets):], np.array([shifted])+1)
	return (scores, rcounts, moves, allscores)

def testCase(n, memory):
	c = n.calculate(memory);
	print(f"Net would run {bcolors.WARNING}{bcolors.BOLD}{np.argmax(c[-1])+1}{bcolors.ENDC} given {bcolors.WARNING}{memory}{bcolors.ENDC}; probabilities {bcolors.WARNING}{list(c[-1])}{bcolors.ENDC}")
	print(f"{bcolors.GREY}{c}{bcolors.ENDC}")

# Fake brain that outputs random output.
class Bot:
	def __init__(self, count, distr):
		self.count = count;
		self.distr = distr;
	@classmethod
	def random(brain, count):
		return brain(count, np.random.dirichlet(np.ones(count)/(10*np.random.sample()**4)));
	def result(self, *args):
		return int(np.random.choice(range(self.count), size=1, p=self.distr));

class Brain:
	def __init__(self, shape, biases, weights):
		self.shape = shape
		self.biases = biases
		self.weights = weights
		self.age = 0;
		self.uuid = int(np.random.sample()*1e9);
		self.reset();
	@classmethod
	def random(brain, shape):
		return brain(shape, [np.random.randn(x) for x in shape[1:]], [np.random.randn(a, b) for a, b in zip(shape[1:], shape[:-1])])

	def reset(self):
		self.plays = 0;
		self.score = 0;
		self.rcount = np.zeros(self.shape[-1]);
		# self.memory = np.zeros(int(self.shape[0]/self.shape[-1]));
	def updateScores(self, scores, rcount):
		self.score+=scores[0];
		self.plays+=scores[1]
		self.rcount+=rcount
	def calculate(self, memory):
		l = np.zeros(self.shape[0])
		for n in range(len(memory)):
			if memory[n] > 0:
				l[int(self.shape[-1]*n+memory[n]-1)] = 1;
		layers = [l]
		for a, b in zip(self.weights, self.biases):
			layers.append((a @ layers[-1])+b) # calculate next layer {sigmoid(weights * layer + biases)}
		# print(layers)
		layers.append(utils.sigmoid(layers[-1]));
		if sum(layers[-1])==0: layers[-1] = np.ones(self.shape[-1]); # if due to precision both are 0
		layers[-1] = layers[-1]/sum(layers[-1]);
		return layers
	def result(self, memory):
		l = int(np.random.choice(range(self.shape[-1]), size=1, p=self.calculate(memory)[-1]));
		return l; # return chosen choice
	def reproduce(self):
		self.age+=1;
		if np.random.sample() < random_chance:
			return Brain.random(self.shape);
		return Brain(
					self.shape,
					[np.vectorize(lambda x:mutateNode(x))(self.biases[i]) for i in range(len(self.shape)-1)],
					[np.vectorize(lambda x:mutateNode(x))(self.weights[i]) for i in range(len(self.shape)-1)]
				)