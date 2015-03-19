# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

import numpy
import settings


def similarity(filename):
    '''Computes the frame to frame similarity
    
    Argument :
        filename: filename of the file containing selected features located in settings.DIR_SELECTED_FEATURES
            (without path, without npy extension)
   
    Returns: 0 if success
    '''
    
    selectedFeatures = numpy.load(settings.DIR_SELECTED_FEATURES + filename + '.npy')
    
    nFeatures, timeSize = selectedFeatures.shape
    
    pastFeatures = selectedFeatures[:,1:]
    futureFeatures = selectedFeatures[:,:-1]
    innerProduct = abs(pastFeatures*futureFeatures).sum(0)
    pastTimeNorms = abs(pastFeatures*pastFeatures).sum(0)
    futureTimeNorms = abs(futureFeatures*futureFeatures).sum(0)
    cosinusTime = innerProduct/numpy.sqrt(pastTimeNorms*futureTimeNorms)
    #plt.xticks(range(0,120,5))
    #plt.plot(numpy.arange(timeSize-1)*(120./(timeSize-1)), cosinusTime)
   
    return cosinusTime

