# Matteo Del Vecchio - Uni-ID: 3885057

from utils import sortByP, getSampledIndex


##### Exercise 2 (b) for Bigram #####

# Bigram model to generate text
def bigram(bigramData):
	bigramSorted = sortByP(bigramData)
	endSentence = False
	result = list()
	chosenWord = '<sos>'
	while not endSentence:
		chosenWord = bigramSample(bigramSorted, chosenWord)
		if chosenWord == '<eos>':
			endSentence = True
		result.append(chosenWord)
	return result

##### END Exercise 2 (b) for Bigram #####


##### Exercise 2 (a) for Bigram #####

# Sampling function for bigram, based on the general one
def bigramSample(bigramData, lastWord):
	# Filtering to consider only bigrams with preceding word
	bigramData = list(filter(lambda x: x[0][0] == lastWord, bigramData))
	chosenIndex = getSampledIndex(bigramData)
	return bigramData[chosenIndex][0][1]

##### END Exercise 2 (a) for Bigram #####
