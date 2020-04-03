Matteo Del Vecchio
Assignment 1 - Statistical Natural Language Processing


Exercise 1
a) See preprocessing() and treatPunctuation() in utils.py
b) See main() in main.py
c) The parameters represent possible combination of different word sequences to keep track of and their number grows exponentially in relation to the number of words in every combination; in other words, with the size of the n-gram.
As soon as the number of words increases, also the number of their combinations increases, in an exponential way.

Exercise 2
a) See getSampledIndex() in utils.py for the general function, unigramSample() in unigram.py, bigramSample() in bigram.py and trigramSample() in trigram.py
b) See unigram(), bigram() and trigram() function in the respective files
c) Bigram and Trigram models generated more reasonable sentences than Unigram model. This is due to the fact that Unigram actually picks every word almost randomly to generate the text while Bigram and Trigrams consider also preceding words. So, assuming the input corpus is grammatically well formed, it is more likely to generate reasonable sentences because n-grams also "encapsulate" correct behaviour of words usage, in some way.


How To Use The Code
The main file is main.py; you have to launch it to generate Uni/Bi/Trigram sentences. Python version used for writing the program is 3.6.5; it will not work if executed with Python 2.x.
The main.py file already includes the indication for being executed with python3 so it is just necessary to change the current directory to the one containing assignment files and launch it with ./main.py.
In case there are problems while launching the program, please check that all .py files have reading and executing permissions.
