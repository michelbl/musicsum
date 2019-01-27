import numpy
import settings
import similarity
import groundtruth
from sklearn import metrics
from sklearn.cluster import KMeans
import new

def potential_states(filename):
    
    similarSegmentsInd = numpy.load(settings.DIR_SEGMENTS_INDICES + filename + '.npy')   
    selectedFeatures = numpy.load(settings.DIR_SELECTED_FEATURES + filename + '.npy')
    
    nFeatures, timeSize = selectedFeatures.shape
    
    nSegments = similarSegmentsInd.shape[0] + 1
    
    potentialStates = numpy.zeros((nFeatures,nSegments))
    potentialStates[:,0] = abs(selectedFeatures[:,:similarSegmentsInd[0]]).sum(1) / similarSegmentsInd[0]
    potentialStates[:,-1] = abs(selectedFeatures[:,similarSegmentsInd[-1]:]).sum(1) / len(range(similarSegmentsInd[-1],timeSize))
    
    for i in range(1,nSegments-1):
        firstInd = similarSegmentsInd[i-1]
        lastInd = similarSegmentsInd[i]
        nFrames = lastInd - firstInd
        potentialStates[:,i] = abs(selectedFeatures[:,firstInd:lastInd]).sum(1) / nFrames
        
    numpy.save(settings.DIR_POTENTIAL_STATES + filename + '.npy', potentialStates)
    
    return potentialStates

def initial_states(filename,threshold):
    potentialStates = numpy.load(settings.DIR_POTENTIAL_STATES + filename + '.npy')
    nFeatures, nPotentialStates = potentialStates.shape
    #threshold = settings.THRESHOLD_INITIAL_STATES
    initialStates = numpy.zeros((nFeatures,1))
    
    while True:
        nPotentialStates = potentialStates.shape[1]
        a = potentialStates[:,0].reshape(nFeatures,1)
        innerProduct = ( potentialStates[:,1:]*numpy.tile(a,(1,nPotentialStates-1)) ).sum(0)
        presentStateNorm = ( potentialStates[:,0] * potentialStates[:,0]).sum(0)
        futureStatesNorms = ( potentialStates[:,1:] * potentialStates[:,1:]).sum(0)
        similarity = innerProduct / numpy.sqrt(presentStateNorm * futureStatesNorms)
        similarStatesInd = numpy.where(similarity > threshold)[0]
        similarStatesInd = similarStatesInd + 1
        similarStatesInd = numpy.concatenate(([0],similarStatesInd))
        
        if similarStatesInd.shape[0] == 1:
            newState = potentialStates[:,0].reshape(nFeatures,1)
        
        else:
            nNewStates = len(similarStatesInd)
            newState = ( potentialStates[:,similarStatesInd] ).sum(1) / nNewStates
            newState = newState.reshape(nFeatures,1)
            
        initialStates = numpy.concatenate((initialStates, newState),axis=1)
        potentialStates = numpy.delete(potentialStates,similarStatesInd,1)
        
        if potentialStates.shape[1] == 0:
            break
        
        if potentialStates.shape[1] == 1:
            initialStates = numpy.concatenate((initialStates, potentialStates),axis=1)
            break
        
    initialStates = numpy.delete(initialStates,0,1)
    
    numpy.save(settings.DIR_INITIAL_STATES + filename + '.npy', initialStates)
    
    return initialStates

def statesfromKmeans(filename):
    initialStates = numpy.load(settings.DIR_INITIAL_STATES + filename + '.npy')
    selectedFeatures = numpy.load(settings.DIR_SELECTED_FEATURES + filename + '.npy')
    initialStates = numpy.transpose(initialStates)
    selectedFeatures = numpy.transpose(abs(selectedFeatures))    
    
    n_clusters, n_features = initialStates.shape
    k_means = KMeans(n_clusters, initialStates,n_init=1, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0)
    kMeansStates = k_means.fit_predict(selectedFeatures)
    
    numpy.save(settings.DIR_KMEANS_STATES + filename + '.npy', kMeansStates)
    
    return kMeansStates

def compute_states(filename):
    threshold = 1
    gap = 0.001
    
    groundTruthSequence = groundtruth.ground_truth_sequence(filename)
    
    while True:
        initialStates = initial_states(filename,threshold)
        kMeansStates = statesfromKmeans(filename)
        if numpy.max(groundTruthSequence) < numpy.max(kMeansStates):
            threshold = threshold - gap
        else:
            old_threshold = threshold
            threshold = threshold + gap / 2
            break
        
    while True:
        initialStates = initial_states(filename,threshold)
        kMeansStates = statesfromKmeans(filename)
        if numpy.max(groundTruthSequence) < numpy.max(kMeansStates):
            gap = abs( old_threshold - threshold )
            old_threshold = threshold
            threshold = threshold - gap /2
        if numpy.max(groundTruthSequence) > numpy.max(kMeansStates):
            gap = abs( old_threshold - threshold )
            old_threshold = threshold
            threshold = threshold + gap / 2
        if numpy.max(groundTruthSequence) == numpy.max(kMeansStates):
            break
    
    print threshold
    
    return kMeansStates, groundTruthSequence

def arrange_states(kMeansStates, groundTruthStates):
    
    arrangedStates = 0 * groundTruthStates
    free_grd_states = range(numpy.max(groundTruthStates)+1)
    
    for i in range(numpy.max(groundTruthStates)+1):
        x = numpy.where(kMeansStates == i,1,0)
        max_common = 0
        for j in free_grd_states:
            y = numpy.where(groundTruthStates == j,1,0)
            common = (x * y).sum()
            if common >= max_common:
                arg_max_common = j
                max_common = common
        ind = numpy.where(kMeansStates == i)
        ind = ind[0]
        arrangedStates[ind] = arg_max_common
        if len(free_grd_states) > 1:
            free_grd_states = numpy.delete(free_grd_states,numpy.where(free_grd_states==arg_max_common),0)
        
    return arrangedStates