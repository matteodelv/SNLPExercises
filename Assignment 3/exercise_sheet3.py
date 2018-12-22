#!/usr/bin/env python3

# Matteo Del Vecchio - Uni-ID: 3885057

################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
import numpy as np
import random
import matplotlib.pyplot as plt


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


def setDotProduct(numpyArray, indexes):
    return sum(map(lambda i: numpyArray[i], indexes))


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
    
    activeFeatures = None
    emFeatureCounts = None

    trainWordCount = 0
    trainBatchWordCount = 0
    
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

        self.activeFeatures = dict()
        self.emFeatureCounts = dict()
        self.emFeatureBatchCounts = dict()

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
    
    
    
    # Exercise 1 b) ###################################################################
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
        for key in self.feature_indices.keys():
            firstElement = key[0]
            secondElement = key[1]
            featureIndex = self.feature_indices[key]
            if (word == firstElement and label == secondElement) or (prev_label == firstElement and label == secondElement):
                activeFeatures.add(featureIndex)

        self.activeFeatures[afKey] = activeFeatures
        return self.activeFeatures[afKey]        



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
            exp = setDotProduct(self.theta, activeFeatures)
            denom += np.exp(exp)

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
        norm = self.cond_normalization_factor(word, prev_label)
        activeFeatures = self.get_active_features(word, label, prev_label)
        exp = setDotProduct(self.theta, activeFeatures)
        expRes = np.exp(exp)
        return norm * expRes
    
    
    
    
    # Exercise 3 a) ###################################################################
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

        result = np.zeros(len(self.feature_indices))
        activeFeatures = self.get_active_features(word, label, prev_label)
        for index in activeFeatures:
            result[index] = 1

        self.emFeatureCounts[emKey] = result
        return self.emFeatureCounts[emKey]
    
    
    
    
    # Exercise 3 b) ###################################################################
    def expected_feature_count(self, word, prev_label):
        '''
        Compute the expected feature count given a word, the label of the previous word and the parameters of the current model
        (see variable theta)
        Parameters: word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the expected feature count
        '''
        result = np.zeros(len(self.feature_indices))

        for label in self.labels:
            activeFeatures = self.get_active_features(word, label, prev_label)
            prob = self.conditional_probability(word, label, prev_label)
            for featureIndex in activeFeatures:
                result[featureIndex] += prob

        return result

    
    
    
    
    # Exercise 4 a) ###################################################################
    def parameter_update(self, word, label, prev_label, learning_rate):
        '''
        Do one learning step.
        Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                    label: string; the actual label of the selected word
                    prev_label: string; the label of the word at position i-1
                    learning_rate: float
        '''
        self.theta = self.theta + learning_rate * (self.empirical_feature_count(word, label, prev_label) - self.expected_feature_count(word, prev_label))
    
    
    
    
    # Exercise 4 b) ###################################################################
    def train(self, number_iterations, learning_rate=0.1):
        '''
        Implement the training procedure.
        Parameters: number_iterations: int; number of parameter updates to do
                    learning_rate: float
        '''
        for i in range(number_iterations):
            randomSentence = random.choice(self.corpus)
            randomIndex = random.randrange(len(randomSentence))
            word = randomSentence[randomIndex][0]
            label = randomSentence[randomIndex][1]
            prevLabel = randomSentence[randomIndex-1][1] if randomIndex > 0 else "start"
            self.trainWordCount += 1
            self.parameter_update(word, label, prevLabel, learning_rate)
    
    
    
    
    # Exercise 4 c) ###################################################################
    def predict(self, word, prev_label):
        '''
        Predict the most probable label of the word referenced by 'word'
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: string; most probable label
        '''
        probs = np.zeros(len(self.labels))
        labelsList = list(self.labels)
        for i in range(len(self.labels)):
            probs[i] = self.conditional_probability(word, labelsList[i], prev_label)

        return labelsList[np.argmax(probs)]
    


    # Exercise 5 a) ###################################################################
    def empirical_feature_count_batch(self, sentences):
        '''
        Predict the empirical feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the empirical feature count
        '''
        emFeatureCounts = np.zeros(len(self.feature_indices))
        for sentence in sentences:
            for i in range(len(sentence)):
                word = sentence[i][0]
                label = sentence[i][1]
                prevLabel = sentence[i-1][1] if i > 0 else "start"
                emFeatureCounts += self.empirical_feature_count(word, label, prevLabel)

        return emFeatureCounts

    
    
    
    
    # Exercise 5 a) ###################################################################
    def expected_feature_count_batch(self, sentences):
        '''
        Predict the expected feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the expected feature count
        '''
        exFeatureCounts = np.zeros(len(self.feature_indices))
        for sentence in sentences:
            for i in range(len(sentence)):
                word = sentence[i][0]
                prevLabel = sentence[i-1][1] if i > 0 else "start"
                exFeatureCounts += self.expected_feature_count(word, prevLabel)

        return exFeatureCounts
    

    
    # Exercise 5 b) ###################################################################
    def train_batch(self, number_iterations, batch_size, learning_rate=0.1):
        '''
        Implement the training procedure which uses 'batch_size' sentences from to training corpus
        to compute the gradient.
        Parameters: number_iterations: int; number of parameter updates to do
                    batch_size: int; number of sentences to use in each iteration
                    learning_rate: float
        '''
        for i in range(number_iterations):
            randomSentences = random.sample(self.corpus, batch_size)
            for sentence in randomSentences:
                self.trainBatchWordCount += len(sentence)
            self.theta = self.theta + learning_rate * (self.empirical_feature_count_batch(randomSentences) - self.expected_feature_count_batch(randomSentences))


# Exercise 5 c) ###################################################################
def evaluate(corpus):
    '''
    Compare the training methods 'train' and 'train_batch' in terms of convergence rate
    Parameters: corpus: list of list; a corpus returned by 'import_corpus'
    '''
    testSetLength = len(corpus) // 10
    testSet = random.sample(corpus, testSetLength)

    trainingSet = corpus.copy()
    for elem in testSet:
        trainingSet.remove(elem)

    A = MaxEntModel()
    A.initialize(trainingSet)

    B = MaxEntModel()
    B.initialize(trainingSet)

    accuraciesA = list()
    wordNumbersA = list()
    accuraciesB = list()
    wordNumbersB = list()

    for i in range(200):
        A.train(1, 0.2)
        B.train_batch(1, 1, 0.2)

        if i % 5 == 0:
            correctA = 0
            correctB = 0
            totalTestWords = 0

            for sentence in testSet:
                totalTestWords += len(sentence)
                for i in range(len(sentence)):
                    word = sentence[i][0]
                    label = sentence[i][1]
                    prevLabel = sentence[i-1][1] if i > 0 else "start"
                    predictedA = A.predict(word, prevLabel)
                    predictedB = B.predict(word, prevLabel)
                    if predictedA == label:
                        correctA += 1
                    if predictedB == label:
                        correctB += 1

            accuraciesA.append(correctA / totalTestWords)
            accuraciesB.append(correctB / totalTestWords)

            wordNumbersA.append(A.trainWordCount)
            wordNumbersB.append(B.trainBatchWordCount)

    for i in range(2):
        wn = wordNumbersA if i == 0 else wordNumbersB
        acc = accuraciesA if i == 0 else accuraciesB

        plt.plot(wn, acc, color="red")
        xMax = max(wn) + 25
        xMin = min(wn) - 25
        plt.xlim([xMin,xMax])
        plt.ylim([0,1.0])
        plt.xlabel("Number of training words")
        plt.ylabel("Accuracy")
        title = "Model A" if i == 0 else "Model B"
        plt.title(title)
        plt.show()
        plt.clf()


def main():
    corpus = import_corpus('corpus_pos.txt')[:200]
    evaluate(corpus)


if __name__ == '__main__':
    main()
