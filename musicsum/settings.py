# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:59:50 2015

"""

WINDOWS_SEPARATOR = '\\'
LINUX_SEPARATOR = '/'

MICHEL_LINUX_PREFIX = '/home/michel/git/musicsum/musicsum/'
MICHEL_WINDOWS_PREFIX = 'C:\Users\Michel\git\musicsum\musicsum\\'
NICO_PREFIX = 'C:\Users\Nicolas\Documents\MVA\Audio Signal Processing\projet_git_2\musicsum\musicsum\\'


DIR_PREFIX = NICO_PREFIX
SEPARATOR = WINDOWS_SEPARATOR


DIR_SONGS = DIR_PREFIX + "songs" + SEPARATOR
DIR_MEL_FEATURES = DIR_PREFIX + "mel_features" + SEPARATOR
DIR_SAMPLE_RATE = DIR_PREFIX + "sample_rate" + SEPARATOR
DIR_DYNAMIC_FEATURES = DIR_PREFIX + "dynamic_features" + SEPARATOR
DIR_SELECTED_FEATURES = DIR_PREFIX + "selected_features" + SEPARATOR
DIR_SEGMENTS_INDICES = DIR_PREFIX + "segments_indices" + SEPARATOR
DIR_UPPER_DIAGONAL = DIR_PREFIX + "upper_diagonal" + SEPARATOR
DIR_POTENTIAL_STATES = DIR_PREFIX + "potential_states" + SEPARATOR
DIR_INITIAL_STATES = DIR_PREFIX + "initial_states" + SEPARATOR
DIR_KMEANS_STATES = DIR_PREFIX + "kmeans_states" + SEPARATOR
DIR_STATES_SEQUENCE = DIR_PREFIX + "states_sequence" + SEPARATOR

FFT_SIZE = 1
OVERLAP = 2
HOP = FFT_SIZE / OVERLAP
TMAX = 120
THRESHOLD = 0.90