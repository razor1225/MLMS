# -*- coding: utf-8 -*-
# @Author: UnsignedByte
# @Date:   21:41:38, 02-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 23:02:32, 03-Dec-2020

import numpy as np

# https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base
def int2base(x, base):
  if x == 0:
    return [0]

  digits = []

  while x:
    digits.append(int(x % base))
    x = int(x / base)

  return digits

def base10(nums, base): # convert list of nums to base 10
	return sum(nums[i]*(base**i) for i in range(len(nums)))


def sigmoid(x):
  return .5 * (1 + np.tanh(.5 * x))

# https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]