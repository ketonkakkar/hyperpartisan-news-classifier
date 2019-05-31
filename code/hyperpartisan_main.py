##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import argparse
import sys
import numpy as np
from HyperpartisanNewsReader import *
from Relabeler import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from HyperpartisanNewsModel import *

def print_links(args):
    extracter = TextTitleFeatures()
    extracter.process(args.training, args.train_size)

def do_experiment(args):
    labeler = BinaryLabels()
    y = labeler.process(args.labels, args.train_size)
    vocabulary = HNVocab(args.vocabulary, args.vocab_size, args.stop_words)
    extracter = BagOfWordsFeatures(vocabulary)

    classifier = MultinomialNB()
    X, ids = extracter.process(args.training, args.train_size)
    X = X.toarray()
    print("X", X)
    # Handle test data
    if args.test_data:
        classifier.fit(X, y)
        X_test, ids = extracter.process(args.test_data, args.test_size)
        class_probs = classifier.predict_proba(X_test)
    else:
        class_probs = cross_val_predict(classifier, X, y, cv=args.xvalidate, method='predict_proba',verbose=0)
    y_pred = [np.argmax(x) for x in class_probs]
    write_output(args.output_file, class_probs, y_pred, ids)

def write_output(out, class_probs, y_pred, ids):
    for i in range(0, len(ids)):
        out.write(ids[i])
        if y_pred[i] == 1:
            out.write(" true ")
        else:
            out.write(" false ")
        out.write(str(class_probs[i][y_pred[i]]) + "\n")

def relabel(args):
    toggle_models(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=argparse.FileType('rb'), help="Training articles")
    parser.add_argument("labels", type=argparse.FileType('rb'), help="Training article labels")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="Vocabulary")
    parser.add_argument("-l", "--links", type=argparse.FileType('r'), default=sys.stdin, help="A file with all the links", metavar="FILE")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file (after stop words)", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances.", default=None)
    parser.add_argument("--test_size", type=int, metavar="N", help="Only test on the first N instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_data", type=argparse.FileType('rb'), metavar="FILE")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    relabel(args)
    # do_experiment(args)
    # print_links(args)
    for fp in (args.output_file, args.training, args.labels, args.vocabulary): fp.close()
