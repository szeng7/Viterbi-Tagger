#!/usr/bin/env python
import sys
import argparse
import math

all_tags = set([])
vocab = {}
tags_count = {}
poss_tags = {}

tag_seq_count = {}
tag_seq_count_sing = {}

word_tag_count = {}
word_tag_count_sing = {}

test_words = []
test_tags = []

class ProbTuple:
    tag = None
    prob = -float('inf')

    def __init__(self, in_tag, in_prob):
        self.tag = in_tag
        self.prob = in_prob

def tt_backoff(curr):
    return tags_count[curr]/(len(test_words)-1.0)

def prob_tt(curr, prev): 

    lambd = 0.5 + tag_seq_count_sing[prev] #altering to 0.5 gives better results, so we stuck with it
    key = curr + "/" + prev
    num = tag_seq_count.get(key, 0) + lambd*tt_backoff(curr)
    denom = tags_count.get(prev, 0) + lambd
    #how many times the sequence of prev -> curr happens divided by the num of prev tags
    return math.log(float(num)/float(denom))

def tw_backoff(word):
    return (vocab.get(word,0) + 1.0)/(len(vocab) + len(test_words) + 1.0)

def prob_tw(word, curr):

    if word == "###" and curr == "###":
        return 0.0
    lambd = 0.5 + word_tag_count_sing[curr]
    key = word + "/" + curr
    num = word_tag_count.get(key, 0) + lambd*tw_backoff(word)
    denom = tags_count.get(curr, 0) + lambd
    #how many times the word-tag pair appears in the training divided by the num of curr tag
    return math.log(float(num)/float(denom))

def test(test_file_name):
    global test_words
    global test_tags

    f = open(test_file_name)
    test_data = f.readlines()
    
    prob = {} #dict of prob of every state
    backpointers = {} #dict of every state in the best parse with its prev state
    mu = {} #dict of the prob of path leading up to every state

    mu["###/0"] = 0.0
    test_words = [None] * len(test_data)
    test_tags = [None] * len(test_data)

    #must initialize pos 0 so that loops below work in the first iteration
    test_words[0] = "###"
    test_tags[0] = "###"

    for i in range(1, len(test_data)):
        line = test_data[i].rstrip("\n")
        line = line.split("/")
        test_words[i] = line[0]
        test_tags[i] = line[1]

        for curr in poss_tags.get(test_words[i], all_tags):
            for prev in poss_tags.get(test_words[i-1], all_tags):
                p = prob_tt(curr, prev) + prob_tw(test_words[i], curr) #arc prob
                key = prev + "/" + str(i-1)
                m = mu[key] + p #prob of best sequence that ends in curr
                key = curr + "/" + str(i)

                if key not in mu or m > mu[key]: #if it's the best probability, keep track
                    mu[key] = m
                    prob[key] = p
                    backpointers[key] = prev

    #backtrace starting from EOS
    best_parse = [None] * len(test_data)
    best_parse[len(test_data) - 1] = "###"
    total_prob = 0.0

    for i in range(len(test_data)-1, 0, -1):
        key = best_parse[i] + "/" + str(i)
        best_parse[i-1] = backpointers[key]
        total_prob += prob[key]

    corr = 0.0 #correct states
    total = 0.0 #total states

    new_corr = 0.0 #new correct states (haven't seen in training)
    new_total = 0.0 #new states total

    for i in range(len(test_words)):
        if test_words[i] == "###":
            continue #skip the rest
        if test_words[i] in vocab:
            total += 1 
        else: #new word
            new_total += 1

        if test_tags[i] == best_parse[i]:
            if test_words[i] in vocab:
                corr += 1
            else:
                new_corr += 1

    accuracy = 100 * (new_corr + corr) / (new_total + total)
    known = 100 * corr / total
    novel = 0 
    if new_total != 0:
        novel = 100 * new_corr / new_total

    print("Tagging accuracy (Viterbi decoding): {0:.2f}% (known: {1:.2f}% novel: {2:.2f}%)".format(accuracy, known, novel))

    perplexity = math.exp(-total_prob/(len(test_words) - 1))

    print("Perplexity per Viterbi-tagged test word: {0:.3f}".format(perplexity))

def logsumexp(x,y): 
    if y <= x:
        return x + math.log(1 + math.exp(y-x))
    else:
        return y + math.log(1 + math.exp(x-y))

def forward_backward(test_file_name):
    f = open(test_file_name)
    test_data = f.readlines()

    global test_words
    global test_tags

    test_words = [None] * len(test_data)
    test_tags = [None] * len(test_data)

    #must initialize pos 0 so that loops below work in the first iteration
    test_words[0] = "###"
    test_tags[0] = "###"

    alpha = {} #dict holding alpha values
    beta = {} #dict holding beta values
    alpha["###/0"] = 0

    for i in range(1, len(test_data)):
        line = test_data[i].rstrip("\n")
        line = line.split("/")
        test_words[i] = line[0]
        test_tags[i] = line[1]

        for curr in poss_tags.get(test_words[i], all_tags):
            for prev in poss_tags.get(test_words[i-1], all_tags):
                p = prob_tt(curr, prev) + prob_tw(test_words[i], curr)
                key = curr + "/" + str(i)
                key2 = prev + "/" + str(i-1)
                alpha[key] = logsumexp(alpha.get(key, -float('inf')), alpha.get(key2, -float('inf')) + p)
    Z = alpha["###/" + str(len(test_words)-1)]

    beta["###/" + str(len(test_words)-1)] = 0
    prob_list = [ProbTuple(None, -float('inf'))] * len(test_words)

    for i in range(len(test_data)-1, -1, -1):
        for curr in poss_tags.get(test_words[i], all_tags):
            key = curr + "/" + str(i)
            if prob_list[i].prob < (alpha.get(key, -float('inf')) + beta.get(key, -float('inf')) - Z):
                prob_list[i] = ProbTuple(curr, alpha.get(key, -float('inf')) + beta.get(key, -float('inf')) - Z)
            for prev in poss_tags.get(test_words[i-1], all_tags):
                key2 = prev + "/" + str(i-1)
                p = prob_tt(curr, prev) + prob_tw(test_words[i], curr)
                beta[key2] = logsumexp(beta.get(key2, -float('inf')), beta.get(key, -float('inf'))+p)

    corr = 0.0 #correct states
    total = 0.0 #total states

    new_corr = 0.0 #new correct states (haven't seen in training)
    new_total = 0.0 #new states total

    outfile = open("test-output", "w")

    for i in range(len(test_words)):  

        outfile.write(str(test_words[i]) + "/" + str(prob_list[i].tag) + "\n")

        if test_words[i] == "###":
            continue #skip the rest
        if test_words[i] in vocab:
            total += 1 
        else: #new word
            new_total += 1

        if test_tags[i] == prob_list[i].tag:
            if test_words[i] in vocab:
                corr += 1
            else:
                new_corr += 1

    accuracy = 100 * (new_corr + corr) / (new_total + total)
    known = 100 * corr / total
    novel = 0 
    if new_total != 0:
        novel = 100 * new_corr / new_total

    print("Tagging accuracy (posterior decoding): {0:.2f}% (known: {1:.2f}% novel: {2:.2f}%)".format(accuracy, known, novel))

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('training')
    parser.add_argument('test')
    args = parser.parse_args()

    training_file = args.training
    test_file = args.test

    #----------START TRAINING DATA-------------
    #read in training file (word/tag format)
    training_data = open(training_file)
    prev = None

    for line in training_data:
        line = line.rstrip("\n")
        line = line.split("/")

        #add to vocab
        if line[0] not in vocab:
            vocab[line[0]] = 1;
        else:
            vocab[line[0]] += 1;

        #add to tags
        if line[1] not in tags_count:
            tags_count[line[1]] = 1
        else:
            tags_count[line[1]] += 1


        hashed = line[0] + "/" + line[1]

        #add to word/tag 
        if hashed not in word_tag_count:
            word_tag_count[hashed] = 1
            poss_tags.setdefault(line[0], [])
            poss_tags[line[0]].append(line[1])
        else:
            word_tag_count[hashed] += 1

        #check for word/tag singletons
        if line[1] not in word_tag_count_sing:
            word_tag_count_sing[line[1]] = 0

        if word_tag_count[hashed] == 1:
            word_tag_count_sing[line[1]] += 1
        elif word_tag_count[hashed] == 2:
            word_tag_count_sing[line[1]] -= 1

        #add to tag/prev_tag
        if prev != None:
            hashed = line[1] + "/" + prev
            if hashed not in tag_seq_count:
                tag_seq_count[hashed] = 1
            else:
                tag_seq_count[hashed] += 1

            #check for tag/prev_tag singletons
            if prev not in tag_seq_count_sing:
                tag_seq_count_sing[prev] = 0

            if tag_seq_count[hashed] == 1:
                tag_seq_count_sing[prev] += 1
            elif tag_seq_count[hashed] == 2:
                tag_seq_count_sing[prev] -= 1

        if line[1] != "###":
            all_tags.add(line[1])

        prev = line[1]

    #----------END TRAINING CODE-------------

    test(test_file)
    forward_backward(test_file)



if __name__ == "__main__":
    main(sys.argv)