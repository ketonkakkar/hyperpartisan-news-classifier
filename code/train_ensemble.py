#Usage: python3 code/train_ensemble.py ensemble1.txt --train_size 10000 --test_size 10000

import argparse
import sys
import numpy as np
from HyperpartisanNewsEnsemble import *
from utilities import *
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

def do_experiment(args):
    classifier_dict = {'MultinomialNB':MultinomialNB,'Vote':Vote,'Tree':DecisionTreeClassifier}
    train_data = open("/data/semeval/v4/training/articles-training-bypublisher-20181122.parsed.xml",'rb')
    # train_labels = open("/data/semeval/v4/training/ground-truth-training-bypublisher-20181122.xml",'rb')
    train_labels = open("relabeled.xml",'rb')
    '''
    test_data = open("/data/semeval/v4/training/articles-training-byarticle-20181122.parsed.xml",'rb')
    test_labels = open("/data/semeval/v4/training/ground-truth-training-byarticle-20181122.xml",'rb')
    '''
    test_data = open("/data/semeval/v4/validation/articles-validation-bypublisher-20181122.parsed.xml",'rb')
    test_labels = open("/data/semeval/v4/validation/ground-truth-validation-bypublisher-20181122.xml",'rb')
    classifier = classifier_dict[args.classifier]()
    ensemble = HNEnsemble(args.models_file,clf=classifier)
    #if args.classifier != "Vote":
    ensemble.train(train_labels,train_data, args.train_size)
    ensemble.test(test_labels,test_data, args.test_size)
    if args.save != None:
        ensemble.save(args.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("models_file", type=argparse.FileType('r'), help="File containing models in ensemble", default='Vote')
    parser.add_argument("--classifier", type=str, help="classifier to use", default="MultinomialNB")
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances.", default=600000)
    parser.add_argument("--test_size", type=int, metavar="N", help="Only test on the first N instances.", default=100000)
    parser.add_argument("--save", type=str, help="File to save model to.", default=None)

    args = parser.parse_args()
    do_experiment(args)
