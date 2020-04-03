#!/usr/bin/env python3

# Matteo Del Vecchio
# SNLP - Assignment 2

import numpy
from numpy import log
from collections import Counter

numpy.seterr(divide='ignore', invalid='ignore')

################################################################################
## SNLP exercise sheet 2
################################################################################

'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the second layer list contains tuples (token,label);
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
        sentence.append((parts[0].upper(), parts[-1]))
        
    f.close()        
    return sentences
    



# Exercise 1 ###################################################################
'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parameterization of this probability distribuion;
                this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''
def initial_state_probabilities(state, internal_representation):
    return internal_representation.get(state, 0)
    
    
    
    
'''
Implement the matrix of transition probabilities.
Parameters:	from_state: string;
            to_state: string;
            internal_representation: data structure representing the parameterization of the matrix of transition probabilities;
                this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''
def transition_probabilities(from_state, to_state, internal_representation):
    return internal_representation[from_state].get(to_state, 0)
    
    
    
    
'''
Implement the matrix of emmision probabilities.
Parameters:	state: string;
            emission_symbol: string;
            internal_representation: data structure representing the parameterization of the matrix of emission probabilities;
                this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''
def emission_probabilities(state, emission_symbol, internal_representation):
    return internal_representation[state].get(emission_symbol, 0)
    
    
    
    
'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''
def estimate_initial_state_probabilities(corpus):
    initState = dict()

    # Calculating number of occurrences for every initial state
    for sentence in corpus:
        firstWord = sentence[0]
        initState[firstWord[1]] = initState.get(firstWord[1], 0) + 1

    # Calculating probabilities
    for state in initState:
        initState[state] = initState[state] / len(corpus)

    return initState
    
    
    
    
'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''
def estimate_transition_probabilities(corpus):
    tagCounts = dict()
    tagPairCounts = dict()

    # Calculating number of occurrences for every state and for every pair of states, representing current state and following state
    for sentence in corpus:
        for i in range(len(sentence)-1):
            curTuple = sentence[i]
            folTuple = sentence[i+1]
            # Number of occurrences for state (used as denominator)
            tagCounts[curTuple[1]] = tagCounts.get(curTuple[1], 0) + 1
            key = (curTuple[1], folTuple[1])
            # Number of occurrences for pair of states, representing the transition (used as numerator)
            tagPairCounts[key] = tagPairCounts.get(key, 0) + 1

    # Calculating probabilities (data structure is dictionary of dictionaries)
    # Key of outer dictionary represents "from state"
    # Key of inner dictionary represents "to state"
    trProb = dict()
    for tag in tagCounts.keys():
        trProb[tag] = dict()
        filteredPairs = list(filter(lambda x: x[0] == tag, tagPairCounts))
        for (curTag, folTag) in filteredPairs:
            trProb[curTag][folTag] = tagPairCounts.get((curTag, folTag), 0) / tagCounts.get(curTag, 1) 

    return trProb
    
    
    
'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_emission_probabilities(corpus):
    tagCounts = dict()
    wordCounts = dict()

    # Calculating number of occurrences for words and states
    for sentence in corpus:
        for tuple in sentence:
            tagCounts[tuple[1]] = tagCounts.get(tuple[1], 0) + 1
            wordCounts[tuple] = wordCounts.get(tuple, 0) + 1

    # Calculating probabilities
    emProbs = dict()
    for tag in tagCounts.keys():
        emProbs[tag] = dict()
        wordsByTag = list(filter(lambda x: x[1] == tag, wordCounts))
        for tuple in wordsByTag:
            word = tuple[0]
            emProbs[tag][word] = wordCounts[tuple] / tagCounts[tuple[1]]    

    return emProbs
    
    
    
    
# Exercise 2 ###################################################################
''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_smbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_smbols, initial_state_probabilities_parameters, transition_probabilities_parameters, emission_probabilities_parameters):

    # Retrieving list of states
    states = list(transition_probabilities_parameters.keys())
    # Viterbi matrix initialized with zeros with dimension n*m (n = number of states, m = length of sentence)
    viterbiMatrix = [[0 for x in range(len(observed_smbols))] for y in range(len(states))]
    # Data structure to keep track of sequence tagging
    sequenceTagging = [None for x in range(len(observed_smbols))]

    ## INIT ##
    for j in range(len(states)):
        word = observed_smbols[0]
        tag = states[j]
        piTag = initial_state_probabilities(tag, initial_state_probabilities_parameters)
        emProb = emission_probabilities(tag, word, emission_probabilities_parameters)
        viterbiMatrix[j][0] = log(piTag) + log(emProb)

    ## INDUCTION ##
    for i in range(1, len(observed_smbols)):
        for j in range(len(states)):
            maxList = list()
            for k in range(len(states)):
                word = observed_smbols[i]
                fromState = states[k]
                toState = states[j]
                trProb = transition_probabilities(fromState, toState, transition_probabilities_parameters)
                emProb = emission_probabilities(toState, word, emission_probabilities_parameters)
                delta = viterbiMatrix[k][i-1]
                maxList.append(delta + log(trProb) + log(emProb))
            viterbiMatrix[j][i] = max(maxList)

    ## TOTAL ##
    # Backward calculation of argmax to retrieve the sequence tagging
    for i in range(len(observed_smbols), 0, -1):
        column = [viterbiMatrix[k][i-1] for k in range(len(states))]
        sequenceTagging[i-1] = states[column.index(max(column))]

    return sequenceTagging


# This function handles corpus preprocessing in which number of occurrences of words gets calculated
# and, in case they appear only once, they are replaced with the <unknown> token
def unknownPreprocessing(corpus):
    counts = Counter()
    for sentence in corpus:
        words = list(map(lambda tuple: tuple[0], sentence))
        counts += Counter(words)

    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            if counts[corpus[i][j][0]] == 1:
                corpus[i][j] = ('<unknown>', corpus[i][j][1])

    return corpus


# This function handles the sentence preprocessing so that it can be used as a test sentence
# Input format of the sentence could be a list of string or a list of tagged tuples (like the one in the corpus)
def testPreprocessing(corpus, testSentence, isPlainString):
    wordSet = set()
    newWordsSet = set()

    for sentence in corpus:
        words = list(map(lambda tuple: tuple[0], sentence))
        wordSet.update(words)

    testWordsList = testSentence
    if not isPlainString:
        testWordsList = list(map(lambda tuple: tuple[0], testSentence))

    newWordsSet.update(testWordsList)
    diffSet = newWordsSet - wordSet

    # if test sentence contains words not appearing in the corpus
    # they are replaced with <unknown>
    if len(diffSet) > 0:
        for i in range(len(testWordsList)):
            if testWordsList[i] in diffSet:
                testWordsList[i] = "<unknown>"

    return testWordsList


def main():
    corpus = import_corpus('./corpus_ner.txt')

    # First string of the corpus is removed and treated as test sentence
    trainCorpus = corpus[1:]
    testSentence = corpus[0]

    trainCorpus = unknownPreprocessing(trainCorpus)
    testSentence = testPreprocessing(trainCorpus, testSentence, False)

    eInitProbs = estimate_initial_state_probabilities(trainCorpus)
    eTrProbs = estimate_transition_probabilities(trainCorpus)
    eEmProbs = estimate_emission_probabilities(trainCorpus)

    tagSequence = most_likely_state_sequence(testSentence, eInitProbs, eTrProbs, eEmProbs)

    print(testSentence)
    print(tagSequence)


if __name__ == '__main__':
    main()

