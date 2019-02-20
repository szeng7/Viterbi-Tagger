## Viterbi Tagging

This is an implementation of Viterbi Tagging for part of speeches of sentences. This is a method that is used to find the most likely sequence of hidden states (path) to obtaining an observed set of results. In this case, it would be predicting the most likely parts of speeches given some sentences. 

Viterbi_tagger is an implementation of the above with both results from the Viterbi Tagging but also the Forward Backward algorithm which finds the most likely probability/designation for each individual state rather than looking at the sequence as a whole. 

Viterbi_em is an implementation with the Expectation Maximization algorithm where the perplexity is decreased (which is good) and the accuracy of the tagging is more accurate through running multiple iterations and using the information from the previous iterations to make better decisions on the next iteration. 

I have included a training file of words and their parts of speeches as well as a test file for testing the accuracy. 

Example Usage:

"""
./viterbi_tagger train test
"""
