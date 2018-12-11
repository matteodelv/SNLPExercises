#!/usr/bin/env python3

# Matteo Del Vecchio - Uni-ID: 3885057

################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
import numpy as np


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




class MaxEntModel(object):
    # training corpus
    corpus = None
    
    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta = None
    
    # dictionary containing all possible features of a corpus and their corresponding index;
    # has to be set by the method 'initialize'; hint: use a Python dictionary
    feature_indices = None
    
    # set containing a list of possible lables
    # has to be set by the method 'initialize'
    labels = None
    
    
    # Exercise 1 a) ###################################################################
    def initialize(self, corpus):
        '''
        Initialize the maximun entropy model, i.e., build the set of all features, the set of all labels
        and create an initial array 'theta' for the parameters of the model.
        Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
        '''
        self.corpus = corpus
        
        wordSet = set()
        self.labels = set()

        # Getting X and Y sets (words and tags)
        for sentence in self.corpus:
            words = list(map(lambda tuple: tuple[0], sentence))
            labels = list(map(lambda tuple: tuple[1], sentence))
            wordSet.update(words)
            self.labels.update(labels)

        # Creating feature set as cartesian product between X and Y (word|tag) and also between Y and Y (tag|tag)
        featureSet = set()
        emissionFeatures = [(word, tag) for word in wordSet for tag in self.labels]
        transitionFeatures = [(tag1, tag2) for tag1 in self.labels for tag2 in self.labels]

        featureSet.update([("start", tag) for tag in self.labels])
        #self.labels.update(["start"])

        featureSet.update(emissionFeatures)
        featureSet.update(transitionFeatures)

        # Initializing feature_indices dictionary assigning an index to every feature
        index = 0
        self.feature_indices = dict()
        for feature in featureSet:
            self.feature_indices[feature] = index
            index += 1

        # Initializing model parameters as numpy array of |F| ones
        self.theta = np.ones(len(self.feature_indices))

        print("Numeber of features = ", len(self.feature_indices))
        print(self.feature_indices)
    
    
    
    # Exercise 1 b) ###################################################################
    def get_active_features(self, word, label, prev_label):
        '''
        Compute the vector of active features.
        Parameters: word: string; a word at some position i of a given sentence
                    label: string; a label assigned to the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing only zeros and ones.
        '''
        activeFeatures = np.zeros(len(self.feature_indices))
        for key in self.feature_indices.keys():
            firstElement = key[0]
            secondElement = key[1]
            activeFeatures[self.feature_indices[key]] = 1 if (word == firstElement and label == secondElement) or (prev_label == firstElement and label == secondElement) else 0
            
        return activeFeatures        



    # Exercise 2 a) ###################################################################
    def cond_normalization_factor(self, word, prev_label):
        '''
        Compute the normalization factor 1/Z(x_i).
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''
        denom = 0.0
        for label in self.labels:
            activeFeatures = self.get_active_features(word, label, prev_label)
            denom += np.exp(np.dot(self.theta, activeFeatures))

        return 1/denom
    
    
    
    # Exercise 2 b) ###################################################################
    def conditional_probability(self, word, label, prev_label):
        '''
        Compute the conditional probability of a label given a word x_i.
        Parameters: label: string; we are interested in the conditional probability of this label
                    word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''
        print("Cond prob")
        norm = self.cond_normalization_factor(word, prev_label)
        exp = np.exp(np.dot(self.theta, self.get_active_features(word, label, prev_label)))
        print(norm)
        print(exp)
        print(norm * exp)
        return norm * exp
    
    
    
    
    # Exercise 3 a) ###################################################################
    def empirical_feature_count(self, word, label, prev_label):
        '''
        Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
        Parameters: word: string; a word x_i some position i of a given sentence
                    label: string; the actual label of the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the empirical feature count
        '''
        print(self.get_active_features(word, label, prev_label))
    
    
    
    
    # Exercise 3 b) ###################################################################
    def expected_feature_count(self, word, prev_label):
        '''
        Compute the expected feature count given a word, the label of the previous word and the parameters of the current model
        (see variable theta)
        Parameters: word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the expected feature count
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 4 a) ###################################################################
    def parameter_update(self, word, label, prev_label, learning_rate):
        '''
        Do one learning step.
        Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                    label: string; the actual label of the selected word
                    prev_label: string; the label of the word at position i-1
                    learning_rate: float
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 4 b) ###################################################################
    def train(self, number_iterations, learning_rate=0.1):
        '''
        Implement the training procedure.
        Parameters: number_iterations: int; number of parameter updates to do
                    learning_rate: float
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 4 c) ###################################################################
    def predict(self, word, prev_label):
        '''
        Predict the most probable label of the word referenced by 'word'
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: string; most probable label
        '''
        
        # your code here
        
        pass
    

def main():
    corpus = import_corpus('prova.txt')

    print(corpus)

    model = MaxEntModel()
    model.initialize(corpus)

    print(model.labels)
    # print(model.get_active_features("b", "q", "q"))
    # print(model.cond_normalization_factor("a", "r"))
    print(model.conditional_probability("b", "q", "q"))
    print(model.empirical_feature_count("a", "q", "start"))


if __name__ == '__main__':
    main()
