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
import timeconv
from matplotlib import pyplot as plt

filenames = ['apocalypse',
    'familyman',
    'fernando',
    'fivemilesout',
    'headoverfeet',
    'heartless',
    'heartofgold',
    'ifieverfeelbetter',
    'ikissedagirl',
    'jaidemandealalune',
    'knowingme',
    'lettreaelise',
    'money',
    'mountteidi',
    'nameofthegame',
    'rimini',
    'shadowontheworld',
    'songtosaygoodbye',
    'sos',
    'stan',
    'timeisrunningout',
    'toninvitation']

'''
# compute mutual information with labeled data to assess quality of features
for filename in filenames:
    print filename
    # compute Mel filterank features
    mel.compute_mfb(filename)
    # compute dynamic features
    dynfeat.compute_dynamic_features(filename)
    selecfeat.mutual_information_features(filename)
'''

for filename in filenames:
    # select a small set of features
    trainFilenames = filenames[:]   # copy the filename list
    trainFilenames.remove(filename)
    selectFeatures = selecfeat.select_features(filename, trainFilenames)

    # compute the full similarity matrix and the upper diagonal (frame to frame distance)
    similarityUpperDiag = similarity.similarity(filename)
    similarityMatrix = similarity.full_similarity(selectFeatures)

    # get the segment dates
    similarSegmentsInd = similarity.segments_indices(filename)

    # plot the upper diagonal
    nPoints = len(similarityUpperDiag)
    rate = numpy.load(settings.DIR_SAMPLE_RATE + filename + '.npy')
    x = timeconv.timeconv(numpy.arange(nPoints), rate, 'feat', 'second')
    plt.plot(x, similarityUpperDiag)
    plt.vlines(timeconv.timeconv(similarSegmentsInd, rate, 'feat', 'second'), min(similarityUpperDiag), 1)
    gtmap = numpy.loadtxt(settings.DIR_LABELS + filename + '.csv', dtype=int, delimiter=',')
    plt.vlines(gtmap[:,0], min(similarityUpperDiag), 1, color='r')
    plt.title('Segmentation based on frame to frame similarity : ' + filename)
    plt.show()
'''
    # plot the matrix
    factor = 100
    plt.imshow(numpy.clip(factor*similarityMatrix-(factor-1), 0, 1), interpolation='nearest')
    plt.title('Frames similarity matrix : ' + filename)
    plt.show()
'''
'''
# compute potential states : mean over segments
potentialStates = states.potential_states(filename)
potentialStatesSimilarityMatrix = similarity.full_similarity(potentialStates)
'''
'''
initialStates = states.initial_states(filename)

kMeansStates = states.statesfromKmeans(filename)

statesSequence = hmm.states_sequence(filename)
'''

'''

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

'''