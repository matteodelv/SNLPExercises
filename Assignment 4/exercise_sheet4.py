#!/usr/bin/env python3

# Matteo Del Vecchio - Uni-ID: 3885057

################################################################################
## SNLP exercise sheet 4
################################################################################
import math
import sys
import numpy as np
from pprint import pprint

'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the first layer list contains the sentences of the corpus;
    the second layer list contains tuples (token,label) representing a labelled sentence
'''
def import_corpus(path_to_file):
    sentences = []
    sentence = []
    f = open(path_to_file)

    while True:
        line = f.readline()
        if not line: break

        line = line.strip()
        if len(line) == 0:
            sentences.append(sentence)
            sentence = []
            continue

        parts = line.split(' ')
        sentence.append((parts[0], parts[-1]))

    f.close()
    return sentences




class LinearChainCRF(object):
    # training corpus
    corpus = None
    
    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta = None
    
    # set containing all features observed in the corpus 'self.corpus'
    # choose an appropriate data structure for representing features
    # each element of this set has to be assigned to exactly one component of the vector 'self.theta'
    features = None
    
    # set containing all lables observed in the corpus 'self.corpus'
    labels = None

    activeFeatures = None
    emFeatureCounts = None
    
    
    def initialize(self, corpus):
        '''
        build set two sets 'self.features' and 'self.labels'
        '''
        self.activeFeatures = dict()
        self.emFeatureCounts = dict()

        self.corpus = corpus
        
        self.features = dict()
        self.labels = set()

        wordSet = set()

        for sentence in self.corpus:
            words = list(map(lambda tuple: tuple[0], sentence))
            labels = list(map(lambda tuple: tuple[1], sentence))
            wordSet.update(words)
            self.labels.update(labels)

        features = set()
        emFeatures = [(word, label) for word in wordSet for label in self.labels]
        trFeatures = [(prevLabel, label) for prevLabel in self.labels for label in self.labels]
        startFeatures = [('start', label) for label in self.labels]

        features.update(emFeatures)
        features.update(trFeatures)
        features.update(startFeatures)

        index = 0
        for feature in features:
            self.features[feature] = index
            index += 1

        self.theta = np.ones(len(self.features))
    
    
    def get_active_features(self, word, label, prev_label):
        '''
        Compute the vector of active features.
        Parameters: word: string; a word at some position i of a given sentence
                    label: string; a label assigned to the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing only zeros and ones.
        '''
        afKey = (word, label, prev_label)
        if self.activeFeatures.get(afKey, None) is not None:
            return self.activeFeatures[afKey]

        activeFeatures = set()
        for key in self.features.keys():
            firstElement = key[0]
            secondElement = key[1]
            featureIndex = self.features[key]
            if (word == firstElement and label == secondElement) or (prev_label == firstElement and label == secondElement):
                activeFeatures.add(featureIndex)

        self.activeFeatures[afKey] = activeFeatures
        return self.activeFeatures[afKey]


    def empirical_feature_count(self, word, label, prev_label):
        '''
        Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
        Parameters: word: string; a word x_i some position i of a given sentence
                    label: string; the actual label of the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the empirical feature count
        '''
        emKey = (word, label, prev_label)
        if self.emFeatureCounts.get(emKey, None) is not None:
            return self.emFeatureCounts[emKey]

        result = np.zeros(len(self.features))
        activeFeatures = self.get_active_features(word, label, prev_label)
        for index in activeFeatures:
            result[index] = 1

        self.emFeatureCounts[emKey] = result
        return self.emFeatureCounts[emKey]


    def psi(self, label, prevLabel, word):
        activeFeatures = self.get_active_features(word, label, prevLabel)
        expArg = sum(map(lambda i: self.theta[i], activeFeatures))
        return np.exp(expArg)


    # Exercise 1 a) ###################################################################
    def forward_variables(self, sentence):
        '''
        Compute the forward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of forward variables
        '''
        labelsList = list(map(lambda tuple: tuple[1], sentence))
        forwardMatrix = [[0 for x in range(len(sentence))] for y in range(len(labelsList))]

        ## INIT
        for j in range(len(labelsList)):
            word = sentence[0][0]
            prevLabel = 'start'
            forwardMatrix[j][0] = self.psi(labelsList[j], prevLabel, word)

        ## INDUCTION
        for i in range(1, len(sentence)):
            for j in range(len(labelsList)): # vanno usate tutte le label oppure solo quelle presenti nella frase in esame????
                word = sentence[i][0]
                label = labelsList[j]
                sum = 0
                for k in range(len(labelsList)):
                    prevLabel = labelsList[k]
                    fi = self.psi(label, prevLabel, word)
                    prevAlpha = forwardMatrix[k][i-1]
                    sum += fi * prevAlpha
                forwardMatrix[j][i] = sum

        return forwardMatrix
  
        
        
    def backward_variables(self, sentence):
        '''
        Compute the backward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of backward variables
        '''
        labelsList = list(map(lambda tuple: tuple[1], sentence))
        backwardMatrix = [[1 for x in range(len(sentence))] for y in range(len(labelsList))]
        
        ## INIT phase is not needed as the matrix is already initialized with ones

        ## INDUCTION
        for i in range(len(sentence)-2, -1, -1):
            for j in range(len(labelsList)):
                word = sentence[i][0]
                prevLabel = labelsList[j]
                sum = 0
                for k in range(len(labelsList)):
                    label = labelsList[k]
                    fi = self.psi(label, prevLabel, word)
                    prevBeta = backwardMatrix[k][i+1]
                    sum += fi * prevBeta
                backwardMatrix[j][i] = sum

        return backwardMatrix
        
        
        
    
    # Exercise 1 b) ###################################################################
    def compute_z(self, sentence):
        '''
        Compute the partition function Z(x).
        Parameters: sentence: list of strings representing a sentence.
        Returns: float;
        '''
        '''
        backwardMatrix = self.backward_variables(sentence)
        labelsList = list(map(lambda tuple: tuple[1], sentence))
        word = sentence[0][0]
        prevLabel = 'start' # non so se è questa... cos'è y0? è riferito alla sentence?

        z = 0
        for i in range(len(labelsList)):
            label = labelsList[i]
            fi = self.psi(label, prevLabel, word)
            betaOne = backwardMatrix[i][0]
            z += fi * betaOne
        '''
        
        forwardMatrix = self.forward_variables(sentence)
        labelsList = list(map(lambda tuple: tuple[1], sentence))
        lastWordIndex = len(sentence)-1

        z = 0
        for i in range(len(labelsList)):
            z += forwardMatrix[i][lastWordIndex]
        
        return z
        
        
        
            
    # Exercise 1 c) ###################################################################
    def marginal_probability(self, sentence, y_t, y_t_minus_one):
        '''
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        Parameters: sentence: list of strings representing a sentence.
                    y_t: element of the set 'self.labels'; label assigned to the word at position t
                    y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
        Returns: float: probability;
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 1 d) ###################################################################
    def expected_feature_count(self, sentence, feature):
        '''
        Compute the expected feature count for the feature referenced by 'feature'
        Parameters: sentence: list of strings representing a sentence.
                    feature: a feature; element of the set 'self.features'
        Returns: float;
        '''
        
        # your code here
        
        pass
    
    
    
    
    
    # Exercise 1 e) ###################################################################
    def train(self, num_iterations, learning_rate=0.01):
        '''
        Method for training the CRF.
        Parameters: num_iterations: int; number of training iterations
                    learning_rate: float
        '''
        
        # your code here
        
        pass
    
    

    
    
    
    
    # Exercise 2 ###################################################################
    def most_likely_label_sequence(self, sentence):
        '''
        Compute the most likely sequence of labels for the words in a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: list of lables; each label is an element of the set 'self.labels'
        '''
        
        # your code here
        
        pass



def main():
    corpus = import_corpus('corpus_pos2.txt')

    model = LinearChainCRF()
    model.initialize(corpus)

    matrix = model.forward_variables(corpus[0])
    pprint(matrix)

    matrix = model.backward_variables(corpus[0])
    pprint(matrix)

    zeta = model.compute_z(corpus[0])
    print(zeta)
    # print(model.features)
    # print()
    # print(model.labels)
    # print()
    # print(model.theta)


if __name__ == '__main__':
    main()
