# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

from features import logfbank
import scipy.io.wavfile as wav
import numpy
import settings


def compute_mfb(filename):
    '''Compute Mel filterbank features on a song and store them in a binary file with the Numpy format.
    
    Argument :
        filename: filename of the wav file located in settings.DIR_SONGS (without path, without wav extension)
    
    Returns: 0 if success
    
    The output file is located in settings.DIR_MEL_FEATURES.
    '''
    tmax = settings.TMAX
    
    (rate,sig) = wav.read(settings.DIR_SONGS + filename + '.wav')
    
    if rate != 44100:
        print 'Warning : the rate is not 44100.'

    nSamples, nChannels = sig.shape
    if nChannels != 2:
        print 'Warning : the number of channels is not 2.'
    if nSamples > rate*tmax:
        sig = sig[:rate*tmax,:]  # take the 2 first minutes (for memory)
    
    #mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    
    numpy.save(settings.DIR_MEL_FEATURES + filename + '.npy', fbank_feat)
    numpy.save(settings.DIR_SAMPLE_RATE + filename + '.npy', rate)
    
    return 0
