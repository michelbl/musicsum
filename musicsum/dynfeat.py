# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

import numpy
import stft
import settings
from features import logfbank


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
    ch = 10
    power = numpy.zeros((nPoints,1))
    for i in range(nPoints):
        power[i] = numpy.abs(melFeatures[i,ch]*melFeatures[i,ch]).sum()
    plt.xticks(range(0,120,5))
    plt.plot(numpy.arange(nPoints)*(120./nPoints), power)
    '''

    timeSize = stft.stft_time_size(melFeatures[:,0], settings.FFT_SIZE, settings.OVERLAP)
    dynamicFeatures = numpy.zeros((nChannels, settings.FFT_SIZE//2+1, timeSize), dtype=complex)
    for i in range(nChannels):
        dynamicFeatures[i,:,:] = stft.stft(melFeatures[:,i], settings.FFT_SIZE, settings.OVERLAP).T
    
    
    numpy.save(settings.DIR_DYNAMIC_FEATURES + filename + '.npy', dynamicFeatures)
    
    return 0

def compute_dynamic_selected_features(filename):
    
    melFeatures = numpy.load(settings.DIR_MEL_FEATURES + filename + '.npy')
    tmax = settings.TMAX
    
    nPoints, nChannels = melFeatures.shape
    if nChannels != 26:
        print "Warning : 26 channels expected"
    
    nChannelsPerChannel = 13
    
    #timeSize = stft.stft_time_size(melFeatures[:,0], settings.FFT_SIZE, settings.OVERLAP)
    #dynamic_selected_features = numpy.zeros((timeSize, nChannels / 2 * nChannelsPerChannel))
    dynamic_selected_features = []
    
    for i in range(nChannels / 2):
        #dynamic_selected_features[:, i*nChannelsPerChannel:(i+1)*nChannelsPerChannel] = logfbank(melFeatures[:,i],100,settings.FFT_SIZE, settings.FFT_SIZE / settings.OVERLAP)[:,:13]
        A = logfbank(melFeatures[:,i],nPoints/tmax,settings.FFT_SIZE, float(settings.FFT_SIZE) / settings.OVERLAP)[:,:13]
        if i == 0:
            dynamic_selected_features = A
        else:
            dynamic_selected_features = numpy.append(dynamic_selected_features,A,axis=1)
            
    dynamic_selected_features = numpy.transpose(dynamic_selected_features)
    
    nFeatures, timeSize = dynamic_selected_features.shape
    
    featureVar = numpy.sqrt(abs(dynamic_selected_features*dynamic_selected_features).mean(1))
    dynamic_selected_features = dynamic_selected_features/numpy.tile(featureVar.reshape((nFeatures,1)), (1,timeSize))
        
    numpy.save(settings.DIR_SELECTED_FEATURES + filename + '.npy', dynamic_selected_features)   
    return dynamic_selected_features
        