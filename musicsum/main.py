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
import states
import hmm
from matplotlib import pyplot as plt

filename = 'takemeout'

mel.compute_mfb(filename)

#dynfeat.compute_dynamic_features(filename)

#selecfeat.select_features(filename)
#selectedFeatures = numpy.load(settings.DIR_SELECTED_FEATURES + filename + '.npy')

selectedFeatures = dynfeat.compute_dynamic_selected_features(filename)

similarityMatrix = similarity.full_similarity(selectedFeatures)

similarityUpperDiag = similarity.similarity(filename)
similarSegmentsInd = similarity.segments_indices(filename)

potentialStates = states.potential_states(filename)
potentialStatesSimilarityMatrix = similarity.full_similarity(potentialStates)
'''
initialStates = states.initial_states(filename)

kMeansStates = states.statesfromKmeans(filename)

statesSequence = hmm.states_sequence(filename)
'''
#plt.hist(similarityUpperDiag, bins=1000)
nPoints = len(similarityUpperDiag)
tmax = settings.TMAX
plt.xticks(range(0,tmax,5))
plt.plot(numpy.arange(nPoints)*(float(tmax)/nPoints), similarityUpperDiag)
plt.vlines(similarSegmentsInd*(float(tmax)/nPoints),min(similarityUpperDiag),1)
plt.title('Segmentation based on frame to frame similarity')
plt.show()

plt.imshow(similarityMatrix, interpolation='nearest')
plt.title('Frames similarity matrix')
plt.show()

plt.imshow(potentialStatesSimilarityMatrix, interpolation='nearest')
plt.title('Potential states similarity matrix')
plt.show()
'''
plt.plot(numpy.arange(len(kMeansStates)), kMeansStates)
plt.title('K-means states sequence')
plt.show()

plt.plot(numpy.arange(len(statesSequence)), statesSequence)
plt.title('HMM states sequence')
plt.show()

plt.imshow(abs(selectedFeatures), interpolation='nearest')
#plt.plot(numpy.arange(len(statesSequence)), statesSequence)
plt.title('Selected features sequence')
plt.show()
'''