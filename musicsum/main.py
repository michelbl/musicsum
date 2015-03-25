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

filename = 'headoverfeet'

# compute Mel filterank features
mel.compute_mfb(filename)

# compute dynamic features
dynfeat.compute_dynamic_features(filename)

# compute mutual information with labeled data to assess quality of features
selecfeat.mutual_information_features(filename)

# select a small set of features
selectFeatures = selecfeat.select_features(filename)

# compute the full similarity matrix and the upper diagonal (frame to frame distance)
similarityUpperDiag = similarity.similarity(filename)
similarityMatrix = similarity.full_similarity(selectFeatures)

# get the segment dates
similarSegmentsInd = similarity.segments_indices(filename)

# compute potential states : mean over segments
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

factor = 5
plt.imshow(numpy.clip(factor*similarityMatrix-(factor-1), 0, 1), interpolation='nearest')
plt.title('Frames similarity matrix')
plt.show()

plt.imshow(potentialStatesSimilarityMatrix, interpolation='nearest')
plt.title('Potential states similarity matrix')
plt.show()

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
