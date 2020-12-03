# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:	 22:05:55, 02-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 09:41:59, 03-Dec-2020

import numpy as np
import utils

# chance to become a random agent
random_chance = 0.05
# chance to completely change a weight
big_mutate_chance = 0.005;
# chance to mutate a weight by value*small_mutate_prop*randn
small_mutate_chance = 0.01
small_mutate_prop = 0.01;

def mutateNode(n):
	if np.random.sample() < small_mutate_chance:
		n=n*(1+small_mutate_prop*np.random.randn());
	if np.random.sample() < big_mutate_chance:
		n=np.random.randn();
	return n;

class Brain:
	def __init__(self, shape, biases, weights):
		self.shape = shape
		self.biases = biases
		self.weights = weights
		self.score = 0;
		self.rcount = np.zeros(self.shape[-1]);
		self.age = 0;
		self.uuid = int(np.random.sample()*1e9);
	@classmethod
	def random(brain, shape):
		return brain(shape, [np.random.randn(x) for x in shape[1:]], [np.random.randn(a, b) for a, b in zip(shape[1:], shape[:-1])])

	def resetScore(self):
		self.score = 0;
		self.rcount = np.zeros(self.shape[-1]);
		self.age+=1;
	def resetMemory(self):
		self.memory = np.full(self.shape[0], -1);
	def calculate(self):
		layers = [self.memory]
		for a, b in zip(self.weights, self.biases):
			layers.append(utils.sigmoid((a @ layers[-1])+b)) # calculate next layer {sigmoid(weights * layer + biases)}
		# print(layers)
		return layers
	def result(self):
		l = np.argmax(self.calculate()[-1]); # last layer (output layer)
		self.rcount[l]+=1
		return l; # return chosen choice
	def reproduce(self):
		if np.random.sample() < random_chance:
			return Brain.random(self.shape);
		return Brain(
					self.shape,
					[np.vectorize(lambda x:mutateNode(x))(self.biases[i]) for i in range(len(self.shape)-1)],
					[np.vectorize(lambda x:mutateNode(x))(self.weights[i]) for i in range(len(self.shape)-1)]
				)