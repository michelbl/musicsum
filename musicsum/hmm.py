import numpy
import settings
from sklearn import hmm

def states_sequence(filename):

    kMeansStates = numpy.load(settings.DIR_KMEANS_STATES + filename + '.npy')
    n_components = len(numpy.unique(kMeansStates))
    
    selectedFeatures = numpy.load(settings.DIR_SELECTED_FEATURES + filename + '.npy')
    selectedFeatures = numpy.transpose(abs(selectedFeatures))  

    model = hmm.GaussianHMM(n_components, "full")
    model.fit([selectedFeatures])
    statesSequence = model.decode(selectedFeatures, algorithm='viterbi')
    statesSequence = statesSequence[1]
    
    numpy.save(settings.DIR_STATES_SEQUENCE + filename + '.npy', statesSequence)
    
    return statesSequence