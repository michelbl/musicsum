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
import groundtruth
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

    # plot the matrix
    factor = 100
    plt.imshow(numpy.clip(factor*similarityMatrix-(factor-1), 0, 1), interpolation='nearest')
    plt.title('Frames similarity matrix : ' + filename)
    plt.show()


   
filename = 'sos'
    
# compute potential states : mean over segments
potentialStates = states.potential_states(filename)
potentialStatesSimilarityMatrix = similarity.full_similarity(potentialStates)

kMeansStates, groundTruthSequence = states.compute_states(filename)

kMeansStates = states.arrange_states(kMeansStates, groundTruthSequence)

statesSequence = hmm.states_sequence(filename)

statesSequence = states.arrange_states(statesSequence, groundTruthSequence)

'''
plt.imshow(potentialStatesSimilarityMatrix, interpolation='nearest')
plt.title('Potential states similarity matrix')
plt.show()
'''
plt.plot(numpy.arange(len(kMeansStates)), groundTruthSequence, color = 'g')
plt.plot(numpy.arange(len(kMeansStates)), kMeansStates, color = 'b')
plt.axis([0, len(kMeansStates), -1, numpy.max([kMeansStates,groundTruthSequence])+1])
plt.title('K-means states sequence')
plt.show()

plt.plot(numpy.arange(len(groundTruthSequence)), groundTruthSequence, color = 'g')
plt.plot(numpy.arange(len(statesSequence)), statesSequence, color = 'b')
plt.axis([0, len(statesSequence), -1, numpy.max([statesSequence,groundTruthSequence])+1])
plt.title('HMM states sequence')
plt.show()

selectedFeatures = numpy.load(settings.DIR_SELECTED_FEATURES + filename + '.npy')

plt.imshow(abs(selectedFeatures), interpolation='nearest')
#plt.plot(numpy.arange(len(statesSequence)), statesSequence)
plt.title('Selected features sequence')
plt.show()

