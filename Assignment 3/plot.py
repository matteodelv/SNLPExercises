#!/usr/bin/env python3

import matplotlib.pyplot as plt

accuraciesA = list()
with open("accuraciesA.txt") as aA:
    accuraciesA = [float(line.strip()) for line in aA]

accuraciesB = list()
with open("accuraciesB.txt") as aB:
    accuraciesB = [float(line.strip()) for line in aB]

wordNumbersA = list()
with open("wordNumbersA.txt") as wnA:
    wordNumbersA = [float(line.strip()) for line in wnA]

wordNumbersB = list()
with open("wordNumbersB.txt") as wnB:
    wordNumbersB = [float(line.strip()) for line in wnB]

# accuracies = [0.254, 0.491]
# wordNumbers = [25, 517]

#plt.axis("equal")
for i in range(2):
	wn = wordNumbersA if i == 0 else wordNumbersB
	acc = accuraciesA if i == 0 else accuraciesB

	plt.plot(wn, acc, color='red')
	#plt.plot(wordNumbersB, accuraciesB, color='green')
	#plt.scatter([wordNumbers[0]], [accuracies[0]], label="Model A", color="green")
	#plt.scatter([wordNumbers[1]], [accuracies[1]], label="Model B", color="blue")
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
#pp = PdfPages('plot.pdf')
#pp.savefig()
#pp.close()