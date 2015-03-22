import numpy
import settings
import similarity
from sklearn import metrics
from sklearn.cluster import KMeans

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

def initial_states(filename):
    potentialStates = numpy.load(settings.DIR_POTENTIAL_STATES + filename + '.npy')
    nFeatures, nPotentialStates = potentialStates.shape
    threshold = settings.THRESHOLD
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