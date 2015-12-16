'''
Vincent Petrella 
NLP Project
260467117
I hereby state that all work featured in this code was produced by myself
'''

import sys, os, codecs
import sklearn
import numpy as np
import collections
import operator
from sklearn import svm, linear_model, naive_bayes, cross_validation

import feature_extraction as features
import data_loader as loader

THREE_LABELS = 1
FOUR_LABELS = 1

use_sentiment_lexicon = True

# evaluation code
def accuracy(label, predict):
    if len(label) != len(predict):
        print 'Label length != predict length'
    else:
        count = 0
        for i in xrange(len(label)):
            if int(label[i]) == int(predict[i]):
                count += 1
        acc = float(count) / len(label)
        print 'Accuracy %d / %d = %.4f' % (count, len(label), acc)

def train_best_model(Xtrain,Xtest,Ytrain,Ytest):
    
    methodAccuracy = []
    
    print 'Logistic Regression'
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(Xtrain,Ytrain)
    
    YPrime = logreg.predict(Xtest)
    methodAccuracy.append((1,accuracy(Ytest,YPrime)))
    
    print 'Support Vector Machine'
    svc = svm.SVC(kernel='linear',C=1.0)
    svc.fit(Xtrain,Ytrain)
    
    YPrime = svc.predict(Xtest)
    methodAccuracy.append((2,accuracy(Ytest,YPrime)))
    
    print 'Naive Bayes'
    naiveBayes = naive_bayes.GaussianNB()
    naiveBayes.fit(Xtrain,Ytrain)
    
    YPrime = naiveBayes.predict(Xtest)
    methodAccuracy.append((3,accuracy(Ytest,YPrime)))
    
    mostAccurate = sorted(methodAccuracy,key=operator.itemgetter(1), reverse=True)[0][0]
    
    return (logreg,svc,naiveBayes), mostAccurate
 
# Run a series of test
def main():
    print '\nReading Training data sets'
    
    reviews     = loader.load_whole_reviews()
    scale_data  = loader.load_scale_data()
    
    data = []
    labels = []
    for k in reviews.keys():
        data.append(reviews[k])
        labels.append(0 if scale_data[k][2] < 7 else 1 )
    
    print sum(labels)
    print sum(labels)/float(len(labels))
    if use_sentiment_lexicon:
        lexicon = loader.load_maxdiff_lexicon("lexicon/Maxdiff-Twitter-Lexicon")
        X, Y = features.load_features(data,labels,features.process_lexicon(lexicon),restrict_corpus=False)
    else:
        X, Y = features.load_features(data,labels)
        
    print 'Model Validation:\n'
    
    print 'With 2-fold Cross-Validation'
    train_indices, test_indices = cross_validation.KFold(len(Y), n_folds=2, shuffle=True, random_state=None)
    
    print 'Cross-validation 1'
    bestModelCV = train_best_model(X[train_indices[0]],X[test_indices[0]],Y[train_indices[0]],Y[test_indices[0]])
    print 'Cross-validation 2'
    bestModelCV2 = train_best_model(X[train_indices[1]],X[test_indices[1]],Y[train_indices[1]],Y[test_indices[1]])
    
    if bestModelCV2[1] > bestModelCV[1]:
        bestModelCV = bestModelCV2
    
    #Confusion = [[0 for x in range(0,6)] for y in range(0,6)] 
	#for i in range(0,len(Yhat)):
	#	Confusion[Ytest[i]][Yhat[i]] += 1
	#
	#print Confusion
    
if __name__ == '__main__':
    np.set_printoptions(threshold='nan')
    
    main()
    
    