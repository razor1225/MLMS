# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   09:39:42, 04-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 14:01:44, 07-Dec-2020

import numpy as np
import utils
from utils import bcolors
import brain
import os
import re
import matplotlib.pyplot as plt
import matplotlib

# 1607068848885
names = input("Dataset name:")

root = os.path.join(os.path.dirname(__file__), 'results')

if not names:
	names = [x for x in os.listdir(root) if re.match(r'^[0-9]+-completed$', x)]
else:
	names = [names]

for name in names:
	print (f'Loading {name}')
	fpath = os.path.join(root, name);

	# gens = [x for x in os.listdir(os.path.join(fpath, 'raws')) if re.match(r'gen_[0-9]+\.npy', x)];

	netCount, gamesPer, fakeAgents, gameCount, generations, P, M, G, brainShape = np.load(os.path.join(fpath, 'params.npy'), allow_pickle=True)
	print(f'Brain shape was {brainShape}, Memory length {int(brainShape[0]/M/P)}')

	percentiles = [0, 10, 25, 50, 75, 90, 100]
	percstyles = ['-', '--', ':', '-', ':', '--', '-']
	percwidth = [0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5]

	substitutions = 100; # number of generations to read out of total

	scorePercentiles = np.zeros((substitutions,len(percentiles)));
	rcountPercentiles = np.zeros((substitutions,M-1,len(percentiles)))

	grange = range(0, generations, int(generations/substitutions));

	for i in range(substitutions):
		nets = sorted(np.load(os.path.join(fpath, 'raws', f'gen_{grange[i]}.npy'), allow_pickle=True), key=lambda x:x.score/x.plays)
		scorePercentiles[i]=np.percentile([x.score/x.plays for x in nets], percentiles);
		rcountPercentiles[i]=np.apply_along_axis(lambda x:np.percentile(x, percentiles), 1, np.transpose([(lambda k:[k[0]]+[k[i-1]+k[i]for i in range(1,len(k))])(x.rcount/x.plays)[:-1] for x in nets]))
		# rcountPercentiles[i]=np.apply_along_axis(lambda x:np.percentile(x, percentiles), 1, np.transpose([x.rcount/x.plays for x in nets]))

		if i==substitutions-1:
			res = brain.runGame([nets[-1]]*P, 100, M, G);
			print(res[3])


		if i%100==0:
			print(f'Generation {grange[i]} completed.')

	plotgrid = (1,2) # num rows, num cols
	cmap = matplotlib.cm.get_cmap('Set1');

	fig = plt.figure(figsize=(8,4))
	scoreAx = fig.add_subplot(*plotgrid, 1);
	for i in range(len(percentiles)):
		scoreAx.plot(grange, scorePercentiles[:,i], label=f'{percentiles[i]}%', linestyle=percstyles[i], linewidth=percwidth[i], color='black')
	scoreAx.set_xlabel('Generation #')
	scoreAx.set_ylabel('Average score per round')
	scoreAx.set_title('Average scores for percentiles')
	scoreAx.set_xlim(0, generations)
	scoreAx.set_ylim(0, G.max())

	rcountAx = fig.add_subplot(*plotgrid, 2, sharex = scoreAx);
	for i in range(len(percentiles)):
		for j in range(M-1):
			rcountAx.plot(grange, rcountPercentiles[:,j,i], label=f'{percentiles[i]}%', linestyle=percstyles[i], linewidth=percwidth[i], color=cmap(j))
	scoreAx.get_legend_handles_labels()
	l1 = rcountAx.legend(*scoreAx.get_legend_handles_labels(), bbox_to_anchor=(1.05, 0, 0.25, 1), loc='upper left', borderaxespad=0.)
	rcountAx.legend(handles=[matplotlib.patches.Patch(color=cmap(i), label=f'Move {i+1}') for i in range(M-1)], bbox_to_anchor=(1.05, 0, 0.25, 1), loc='lower left', borderaxespad=0.)
	rcountAx.add_artist(l1)
	rcountAx.set_title('Average proportions of each move')
	rcountAx.set_ylabel('Proportion')
	rcountAx.set_xlabel('Generation #')
	rcountAx.set_ylim(0, 1)

	plt.savefig(os.path.join(fpath, 'scores.svg'), bbox_inches='tight')

	# print(f"\n{bcolors.HEADER}Best brain for generation {generations-1}{bcolors.ENDC}")
	# n = nets[np.argmax(np.vectorize(lambda x:x.score/x.plays)(nets))]
	# print(sum([i.rcount for i in nets]))
	# print(f'{bcolors.OKGREEN}Average score per game: {n.score/n.plays}{bcolors.ENDC}')
	# print(f'Number of each choice: {bcolors.FAIL}{n.rcount}{bcolors.ENDC}')

	# for m in brain.cases:
	# 	if len(m) == brainShape[0]/M:
	# 		brain.testCase(n, m)
