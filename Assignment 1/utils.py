# Matteo Del Vecchio

import random

##### Exercise 1 (a) #####

# Function that handles punctuation symbols by removing them
# or replacing them with start/end sentence markers
def treatPunctuation(line):
	toBeStripped = ["\n", ",", ";", ":", "-"]
	for v in toBeStripped:
		line = line.replace(v, "")
	endSentences = ["?", "!", "."]
	for v in endSentences:
		line = line.replace(v, " <eos> <sos>")
	return line


# Main preprocessing function
def preprocessing(line):
	line = "<sos> " + line
	line = treatPunctuation(line).strip()
	if line.endswith(" <sos>"):		# removes last <sos>
		line = line[:-6]
	words = line.split(" ")
	words = list(map(lambda word: word.lower(), words))		# transform each word to lowercase
	return words

##### END Exercise 1 (a) #####


# Utility function to sort dictionary by probability values, in reverse order
def sortByP(d):
	return sorted(d.items(), key=lambda x: x[1], reverse=True)


##### Exercise 2 (a) #####

# General sampling function: from a multinomial distribution
# it draws the index for the chosen word
# data is a list of tuples (key, prob value)
def getSampledIndex(data):
	outcomeCount = len(data)
	chosenIndex = -1
	probSum = 0
	randomX = random.uniform(0.0, 1.0)
	for i in range(0, outcomeCount):
		probSum += data[i][1]
		if probSum - randomX >= 0:
			chosenIndex = i
			break
	return chosenIndex

##### END Exercise 2 (a) #####
