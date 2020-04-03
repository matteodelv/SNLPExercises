# Matteo Del Vecchio

from utils import sortByP, getSampledIndex
from bigram import bigramSample


##### Exercise 2 (b) for Trigram #####

# Trigram model to generate text
def trigram(trigramData, bigramData):
	trigramSorted = sortByP(trigramData)
	bigramSorted = sortByP(bigramData)
	endSentence = False
	result = list()
	chosenTuple = ('<sos>', '<sos>')
	while not endSentence:
		# Start case
		if chosenTuple == ('<sos>', '<sos>'):
			chosenWord = bigramSample(bigramSorted, chosenTuple[0])
		# General case
		else:
			chosenWord = trigramSample(trigramSorted, chosenTuple)
		# Tuple used to keep track of preceding two words after shifting them for the next iteration
		chosenTuple = (chosenTuple[1], chosenWord)
		if chosenWord == '<eos>':
			endSentence = True
		result.append(chosenWord)
	return result

##### END Exercise 2 (b) for Trigram #####


##### Exercise 2 (a) for Trigram #####

# Sampling function for trigram, based on general one
def trigramSample(trigramData, lastTuple):
	# Filtering to consider only trigrams with preceding two words
	trigramData = list(filter(lambda x: x[0][0] == lastTuple[0] and x[0][1] == lastTuple[1], trigramData))
	chosenIndex = getSampledIndex(trigramData)
	return trigramData[chosenIndex][0][2]

##### END Exercise 2 (a) for Trigram #####
