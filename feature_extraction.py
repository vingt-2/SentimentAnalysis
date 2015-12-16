'''
Vincent Petrella 
NLP Project
260467117
I hereby state that all work featured in this code was produced by myself
'''
import sys, codecs
import numpy as np
import collections
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.util import *
from nltk.probability import FreqDist

STOPWORDS = set(stopwords.words('english'))

filterStopwords = True
lowercaseTokens = True
lemmatizeTokens = True
considerBigrams = False

most_frequent = 0

def load_features(texts,values,sentiment_lexicon=None,restrict_corpus=False):
    bows = []
    
    for lines in texts:
        text = ''
        for l in lines:
            text += l
        bows.append(process_tokens(get_tokens(text),sentiment_lexicon,restrict_corpus))

   
    #print 'Creating High Dimensional Feature Matrix:'
    if sentiment_lexicon:
        X = collect_features(bows,sentiment_lexicon)
    else:
        X = collect_features(bows)    
    print '.. %i features for %i documents' % (len(X[0]),len(X))
    
    if most_frequent != 0:
        print 'Reducing feature dimension to the %i most frequent features' % most_frequent
        X = reduce_features(X, most_frequent)

    Y = np.array(values)
    X = np.vstack(tuple(X))
    
    return X, Y

def is_legal(s):
    return (all(char.isalpha() for char in s) and (len(s) > 1))

def get_tokens(s):
    retval = []
    sents = sent_tokenize(s)

    for sent in sents:
        tokens = word_tokenize(sent)
        retval.extend(tokens)
    return retval

def process_lexicon(lexicon):
    result = dict()
    lemmatizer = WordNetLemmatizer()
    for k,v in lexicon.items():
        a = lemmatizer.lemmatize(k.lower())
        result[a] = v
        
    return result
    
def process_tokens(tokens,sentiment_lexicon=None,restrict_corpus=False):

    copy = tokens
    tokens = []
    for w in copy:
        subwords = w.split('_')
        tokens.extend(subwords)
    
    if restrict_corpus:
        tokens = [w for w in tokens if is_legal(w)]
    else:
        tokens = [w for w in tokens if is_legal(w)]
    
    if lowercaseTokens:
        #All to lowercase
        tokens = [w.lower() for w in tokens]
    
    if filterStopwords:
        #Remove Stopwords
        tokens = [w for w in tokens if w not in STOPWORDS]
    
    if lemmatizeTokens:
        #lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    if restrict_corpus:
        tokens = list(set(tokens) & set(sentiment_lexicon.keys()))
    
    if considerBigrams:
        tokens.extend(list(bigrams(tokens)))
    
    return FreqDist(tokens)

# Collect features from lists of words. Build a corpus if needed
def collect_features(bows,sentiment_lexicon=None):
    X = []
    corpus = []
 
    sys.stdout.write("-Collecting Corpus")
    i = 0
    for bow in bows:
        if (1000*i/len(bows)) % 100 == 0: 
            sys.stdout.write(".")
        corpus = list(set(corpus)|set(bow.keys()))
        i = i + 1
    print ' '
    with open('out.txt','w') as file:
        for w in corpus:
            file.write(w.encode('ascii', 'ignore')+'\n')
    
    sys.stdout.write("-Filling up Matrix")
    for r in range(0,len(bows)):
        if (1000*r/len(bows)) % 100 == 0:
            sys.stdout.write(".")
        countList = []
        for word in corpus:
            a = 0.0
            try:
                a = bows[r][word]
            except KeyError:
                b = 10
            if sentiment_lexicon:
                try:
                    a = a * 1000 * abs(sentiment_lexicon[word])
                except KeyError:
                    a = a * 150
            countList.append(a)
        X.append(countList)
    return X

# Keep the n most frequent features 
def reduce_features(X,n):
    columnSums = collections.Counter()
    for c in range(0,len(X[0])):
        total = 0
        for r in range(0,len(X)):
            total += X[r][c]
        columnSums[c] = int(total)

    l = columnSums.most_common(n)

    mostFrequentFeatures = [a for (a,b) in l]
    
    Xt = np.transpose(X)

    filteredXt = []
    for c in range(0,len(X[0])):
        if c in mostFrequentFeatures:
            filteredXt.append(Xt[c])

    return np.transpose(filteredXt)   
