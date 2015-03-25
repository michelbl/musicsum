# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

NB : This file is (in part) a snipet found on stackoverflow.
"""

import scipy
import numpy

def stft_time_size(x, fftsize, overlap):
    hop = fftsize // overlap
    return len(range(0, len(x)-fftsize, hop))


def stft(x, fftsize, overlap):
    '''Computes the Short Time Fourier Transform with sensible defaults : Hanning window, window length is a power of 2 
    '''
    
    hop = fftsize // overlap
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]  
    return numpy.array([numpy.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])


def istft(X, overlap=4):
    # Warning : not tested !
    
    fftsize=(X.shape[1]-1)*2
    hop = fftsize / overlap
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop) 
    for n,i in enumerate(range(0, len(x)-fftsize, hop)): 
        x[i:i+fftsize] += scipy.real(numpy.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x
