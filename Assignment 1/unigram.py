# Matteo Del Vecchio

from utils import sortByP, getSampledIndex


##### Exercise 2 (b) for Unigram #####

# Unigram model that generate text
def unigram(unigramDict):
	sortedDict = sortByP(unigramDict)
	endSentence = False
	result = list()
	while not endSentence:
		chosenWord = unigramSample(sortedDict)
		if chosenWord == '<sos>':
			continue
		if chosenWord == '<eos>':
			endSentence = True
		result.append(chosenWord)
	return result

##### END Exercise 2 (b) for Unigram #####


##### Exercise 2 (a) for Unigram #####

# Sampling function for unigram, based on the general one
def unigramSample(uniDict):
	chosenIndex = getSampledIndex(uniDict)
	return uniDict[chosenIndex][0]

##### END Exercise 2 (a) for Unigram #####
