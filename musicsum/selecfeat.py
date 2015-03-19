# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

import numpy
import settings
import itertools


def select_features(filename):
    '''Select the best dynamic features.
    
    Argument :
        filename: filename of the file containing dynamic features located in settings.DIR_DYNAMIC_FEATURES
            (without path, without npy extension)
   
    Returns: 0 if success
    
    The output file is located in settings.DIR_SELECTED_FEATURES.
    '''
    
    dynamicFeatures = numpy.load(settings.DIR_DYNAMIC_FEATURES + filename + '.npy')
        
    nChannels, freqSize, timeSize = dynamicFeatures.shape
    
    '''
    power = numpy.zeros((timeSize,1))
    for i in range(timeSize):
        power[i] = numpy.abs(dynamicFeatures[:,:,i]*dynamicFeatures[:,:,i]).sum()
    plt.plot(numpy.arange(timeSize)*(120./timeSize), power)
    '''

    
    channelSet = range(13)
    freqSet = [pow(2, i)-1 for i in range(int(numpy.log(freqSize)/numpy.log(2))+1)]
    selectedIndexes = itertools.product(channelSet, freqSet)
    
    flatIndexes=itertools.starmap(lambda x,y: x*freqSize+y, selectedIndexes)
    selectedFeatures = dynamicFeatures.reshape(nChannels*freqSize, timeSize)[list(flatIndexes),:]
    nFeatures, timeSize = selectedFeatures.shape
    
    # Normalization/coefficient (very important)
    featureVar = numpy.sqrt(abs(selectedFeatures*selectedFeatures).mean(1))
    #plt.plot(featureVar)
    #coef = featureVar
    coef = numpy.sqrt(featureVar)
    #coef = numpy.ones((nFeatures, 1))
    selectedFeatures = selectedFeatures/numpy.tile(coef.reshape((nFeatures,1)), (1,timeSize))

    
    numpy.save(settings.DIR_SELECTED_FEATURES + filename + '.npy', selectedFeatures)
    
    return 0

