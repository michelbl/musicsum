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

def full_similarity(filename):
    '''Computes the full similarity matrix
    '''
    
    selectedFeatures = numpy.load(settings.DIR_SELECTED_FEATURES + filename + '.npy')
    
    nFeatures, timeSize = selectedFeatures.shape
    
    similarityMatrix = numpy.identity(timeSize)
    for i in range(0,timeSize):
        for j in range(i+1,timeSize):
            pastFeatures = selectedFeatures[:,i]
            futureFeatures = selectedFeatures[:,j]
            innerProduct = abs(pastFeatures*futureFeatures).sum(0)
            pastTimeNorms = abs(pastFeatures*pastFeatures).sum(0)
            futureTimeNorms = abs(futureFeatures*futureFeatures).sum(0)
            similarityMatrix[i,j] = innerProduct/numpy.sqrt(pastTimeNorms*futureTimeNorms)

    return similarityMatrix


def segments(similarityUpperDiag):
    '''Detects the segments (segments are cut were the similarity falls below settings.THRESHOLD)
    '''
    
    threshold = settings.THRESHOLD
    '''elements are 1 where the similarity is above threshold, -1 otherwise'''
    binary = numpy.where(similarityUpperDiag > threshold,1,-1)
    segmentsIndices = 0*similarityUpperDiag
    segmentsIndices[0:-1] = binary[0:-1]*binary[1:]
    segmentsIndices = numpy.where(segmentsIndices < 0)
    segmentsIndices = segmentsIndices[0]
    
    '''numpy.save(settings.DIR_SEGMENTS_INDICES + filename + '.npy', segmentsIndices)'''
    
    return segmentsIndices