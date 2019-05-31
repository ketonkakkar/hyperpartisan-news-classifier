#Usage: python3 code/train_model.py --vocabulary /data/semeval/training/vocab.txt --xvalidate 5 --train_size 1000 --test_size 100
#Usage: python3 code/train_model.py --model Links --vocabulary code/links.txt --vocab_size 1000000 --xvalidate 5 --train_size 600000 --test_size 100000
#Test on by article: python3 code/train_model.py --vocabulary /data/semeval/training/vocab.txt --xvalidate 6 --vocab_size 10000 --test_size 600 --model TitleSentences

import argparse
import sys
import numpy as np
from HyperpartisanNewsReader import *
from HyperpartisanNewsModel import *
#from HyperpartisanNewsTFModel import *
from HyperpartisanNewsKerasModel import *
from utilities import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def do_experiment(args):
    model_dict = {'BagOfWords':BagOfWords, 'Links':Links,
    'TextSentences':TextSentences, 'TextWords':TextWords,
    'TitleSentences':TitleSentences, 'TitleWords':TitleWords}

    train_data = open("/data/semeval/v4/training/articles-training-bypublisher-20181122.parsed.xml",'rb')
    train_labels = open("relabeled.xml",'rb')
    '''
    test_data = open("/data/semeval/v4/validation/articles-validation-bypublisher-20181122.parsed.xml",'rb')
    test_labels = open("/data/semeval/v4/validation/ground-truth-validation-bypublisher-20181122.xml",'rb')
    '''
    test_data = open("/data/semeval/v4/training/articles-training-byarticle-20181122.parsed.xml",'rb')
    test_labels = open("/data/semeval/v4/training/ground-truth-training-byarticle-20181122.xml",'rb')

    if args.reload == None:
        if args.vocabulary != None:
            vocabulary = HNVocab(args.vocabulary, args.vocab_size, args.stop_words)
            model = model_dict[args.model](vocabulary=vocabulary)
        else:
            model = model_dict[args.model]()
        model.train(train_labels,train_data, args.train_size,args.xvalidate)
    else:
        model = model_dict[args.model]()
        model.restore(args.reload)
    model.test(test_labels,test_data, args.test_size)
    if args.save != None:
        model.save(args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name", default="BagOfWords")
    parser.add_argument("--vocabulary", type=argparse.FileType('r'), help="Vocabulary", default=None)
    parser.add_argument("--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=0)
    parser.add_argument("--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file (after stop words)", default=10000)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances.", default=600000)
    parser.add_argument("--test_size", type=int, metavar="N", help="Only test on the first N instances.", default=100000)
    parser.add_argument("--xvalidate", type=int, metavar="N", help="Number of folds for cross validation.", default=None)
    parser.add_argument("--save", type=str, help="File to save model to.", default=None)
    parser.add_argument("--reload", type=str, help="File to load model from.", default=None)

    args = parser.parse_args()
    do_experiment(args)
