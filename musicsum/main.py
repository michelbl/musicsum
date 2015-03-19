# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

import numpy
import settings
import mel
import dynfeat
import selecfeat
import similarity
from matplotlib import pyplot as plt

filename = 'takemeout'

mel.compute_mfb(filename)
dynfeat.compute_dynamic_features(filename)
selecfeat.select_features(filename)
similarityUpperDiag = similarity.similarity(filename)


#plt.hist(similarityUpperDiag, bins=1000)
nPoints = len(similarityUpperDiag)
plt.xticks(range(0,120,5))
plt.plot(numpy.arange(nPoints)*(120./nPoints), similarityUpperDiag)
plt.show()



'''
a = numpy.zeros((3,3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            a[i,j,k] = i + 10*j + 100*k
a.reshape((9,3))[:,0]

a[:,:,0]'''