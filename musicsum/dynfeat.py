# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

import numpy
import stft
import settings


def compute_dynamic_features(filename):
    '''Compute dynamic features given Mel filterbank features.
    
    Argument :
        filename: filename of the file containing Mel filterbank features located in settings.DIR_MEL_FEATURES
            (without path, without npy extension)
   
    Returns: 0 if success
    
    The output file is located in settings.DIR_DYNAMIC_FEATURES.
    '''
    
    melFeatures = numpy.load(settings.DIR_MEL_FEATURES + filename + '.npy')
    
    nPoints, nChannels = melFeatures.shape
    if nChannels != 26:
        print "Warning : 26 channels expected"
    
    '''    
    power = numpy.zeros((nPoints,1))
    for i in range(nPoints):
        power[i] = numpy.abs(melFeatures[i,:]*melFeatures[i,:]).sum()
    plt.xticks(range(0,120,5))
    plt.plot(numpy.arange(nPoints)*(120./nPoints), power)
    '''

    timeSize = stft.stft_time_size(melFeatures[:,0], settings.FFT_SIZE, settings.OVERLAP)
    dynamicFeatures = numpy.zeros((nChannels, settings.FFT_SIZE//2+1, timeSize), dtype=complex)
    for i in range(nChannels):
        dynamicFeatures[i,:,:] = stft.stft(melFeatures[:,i], settings.FFT_SIZE, settings.OVERLAP).T
    
    
    numpy.save(settings.DIR_DYNAMIC_FEATURES + filename + '.npy', dynamicFeatures)
    
    return 0

