# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

import numpy
import settings
import itertools
import timeconv


def select_features(filename, mutinf_filenames):
    '''Select the best dynamic features.
    
    Argument :
        filename: filename of the file containing dynamic features located in settings.DIR_DYNAMIC_FEATURES
            (without path, without npy extension)
   
    Returns: 0 if success
    
    The output file is located in settings.DIR_SELECTED_FEATURES.
    '''
    
    dynamicFeatures = numpy.load(settings.DIR_DYNAMIC_FEATURES + filename + '.npy')
    rate = numpy.load(settings.DIR_SAMPLE_RATE + filename + '.npy')
        
    nChannels, freqSize, timeSize = dynamicFeatures.shape
    
    ''' Plot feature power
    ch = 25
    freq = 100
    power = numpy.zeros((timeSize,1))
    for i in range(timeSize):
        power[i] = numpy.abs(dynamicFeatures[ch,freq,i]*dynamicFeatures[ch,freq,i]).sum()
    plt.plot(numpy.arange(timeSize)*(120./timeSize), power)
    '''

    # load quality information and select the best features
    mutinf = numpy.zeros((nChannels, freqSize))
    for mutinf_f in mutinf_filenames:
        mutinf = mutinf + numpy.load(settings.DIR_MUTINF_FEATURES + mutinf_f + '.npy')
    
    ind = numpy.unravel_index(numpy.argsort(mutinf.flatten())[-settings.N_FEATURES:], mutinf.shape)
    selectedIndexes = [(ind[0][i], ind[1][i]) for i in range(settings.N_FEATURES)]
    
    ''' Select cartesian product of features
    channelSet = range(13)
    freqSet = [pow(2, i)-1 for i in range(int(numpy.log(freqSize)/numpy.log(2))+1)]
    #freqSet = [pow(2, i)-1 for i in range(int(numpy.log(rate/2))+1)]
    selectedIndexes = itertools.product(channelSet, freqSet)
    '''
    
    ''' Select some features (doesn't work well)
    selectedIndexes = [(i, i//2+1) for i in range(20)]
    '''
    
    flatIndexes=itertools.starmap(lambda x,y: x*freqSize+y, selectedIndexes)    # could also use numpy.unravel_index() ...
    selectedFeatures = dynamicFeatures.reshape(nChannels*freqSize, timeSize)[list(flatIndexes),:]
    nFeatures, timeSize = selectedFeatures.shape
    
    # Normalization/coefficient (very important)
    featureSigma = numpy.sqrt(abs(selectedFeatures*selectedFeatures).mean(1))
    #plt.plot(featureVar)
    coef = mutinf[ind]/featureSigma   # normalization to 1
    #coef = numpy.sqrt(featureVar)
    #coef = numpy.ones((nFeatures, 1)) # no normalization
    selectedFeatures = selectedFeatures*numpy.tile(coef.reshape((nFeatures,1)), (1,timeSize))

    
    numpy.save(settings.DIR_SELECTED_FEATURES + filename + '.npy', selectedFeatures)
    
    return selectedFeatures

def mutual_information_features(filename):
    '''Compute mutual information between dynamic features and ground truth labels
    
    Returns: 0 if success
    '''
    
    dynamicFeatures = numpy.abs(numpy.load(settings.DIR_DYNAMIC_FEATURES + filename + '.npy'))
    rate = numpy.load(settings.DIR_SAMPLE_RATE + filename + '.npy')
    nChannels, freqSize, timeSize = dynamicFeatures.shape
    
    # compute ground truth features
    gtmap = numpy.loadtxt(settings.DIR_LABELS + filename + '.csv', dtype=int, delimiter=',')
    nPhases = gtmap[:,1].max() + 1
    nSection = gtmap.shape[0]
    groundTruth = numpy.zeros((nPhases, timeSize))
    for i in range(nSection-1):
        beginning = max(int(round(timeconv.timeconv(gtmap[i, 0], rate, 'second', 'feat'))), 0)
        end = max(int(round(timeconv.timeconv(gtmap[i+1, 0], rate, 'second', 'feat'))), 0)
        if(beginning < timeSize):
            if (end >= timeSize):
                end = -1
            groundTruth[gtmap[i, 1], beginning:end] = 1
    groundTruth = groundTruth[groundTruth.sum(1)>0,:]
    nPhases = groundTruth.shape[0]
    #plt.plot(range(timeSize), groundTruth[0,:], range(timeSize), groundTruth[1,:], range(timeSize), groundTruth[2,:], range(timeSize), groundTruth[3,:], range(timeSize), groundTruth[4,:], )
    
    # compute mutual informations
    num = 10
    mutinf = numpy.zeros((nChannels, freqSize))
    for ch in range(nChannels):
        for freq in range(freqSize):
            for i in range(nPhases):
                bins = numpy.linspace(dynamicFeatures[ch, freq, :].min(), dynamicFeatures[ch, freq, :].max(), num=num)
                d = float(timeSize)
                hist = numpy.histogram(dynamicFeatures[ch, freq, :], bins)[0]/d
                hist1 = numpy.histogram(dynamicFeatures[ch, freq, groundTruth[i,:]==True], bins)[0]/d
                hist0 = numpy.histogram(dynamicFeatures[ch, freq, groundTruth[i,:]==False], bins)[0]/d
                p0 = sum(groundTruth[i,:]==False)/d
                p1 = sum(groundTruth[i,:]==True)/d
                #entFeature = -(hist*numpy.nan_to_num(numpy.log2(hist))).sum()/num
                #entGround = -(p0*numpy.log2(p0) + p1*numpy.log2(p1))
                mutinf[ch, freq] = mutinf[ch, freq] + (hist0*numpy.nan_to_num(numpy.log2(hist0/(hist*p0))) + hist1*numpy.nan_to_num(numpy.log2(hist1/(hist*p1)))).sum()/num
                #plt.plot(mutinf)
                #print(mutinf.sum())
        
    #plt.plot(mutinf.T)
    
  
    numpy.save(settings.DIR_MUTINF_FEATURES + filename + '.npy', mutinf)

    return mutinf

