# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   09:39:42, 04-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 23:18:21, 04-Dec-2020

import numpy as np
import utils
from utils import bcolors
import brain
import os
import re
import matplotlib.pyplot as plt

# 1607068848885
name = input("Dataset name:")

fpath = os.path.join(os.path.dirname(__file__), 'results', name);

# gens = [x for x in os.listdir(os.path.join(fpath, 'raws')) if re.match(r'gen_[0-9]+\.npy', x)];

netCount, gamesPer, fakeAgents, gameCount, generations, P, M, G, brainShape = np.load(os.path.join(fpath, 'params.npy'), allow_pickle=True)

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
	[2,2,2,2,2,2]
]

nets = np.load(os.path.join(fpath, 'raws', f'gen_{generations-1}.npy'), allow_pickle=True)
print(f"\n{bcolors.HEADER}Best brain for generation {generations-1}{bcolors.ENDC}")
n = nets[np.argmax(np.vectorize(lambda x:x.score/x.plays)(nets))]
print(sum([i.rcount for i in nets]))
print(f'{bcolors.OKGREEN}Average score per game: {n.score/n.plays}{bcolors.ENDC}')
print(f'Number of each choice: {bcolors.FAIL}{n.rcount}{bcolors.ENDC}')

for m in cases:
	if len(m) == brainShape[0]/M:
		brain.testCase(n, m)

# fig, ax = plt.subplots();
