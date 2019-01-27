import numpy
import settings
import timeconv

def ground_truth_sequence(filename):
    '''Compute the ground truth labels sequence
    
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
    groundTruthSequence = numpy.argmax(groundTruth, axis = 0)
    
    return groundTruthSequence