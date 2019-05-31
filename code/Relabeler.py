import argparse
import sys
import numpy as np
from HyperpartisanNewsReader import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

def toggle_models(args):
    model_list = [BagOfWordsFeatures, LinksFeatures, EmbededTitleSentencesFeatures]
    feature_dict = {}
    X_list = []
    id_list = []
    classifier_list = []
    for model in model_list:
        X, ids, classifier = cross_pub_val(args, model)
        X_list.append(X)
        id_list.append(ids)
        classifier_list.append(classifier)
    for i in range(2):
        for j, model in enumerate(model_list):
            id_dict = make_labels(args, X_list[j], id_list[j], classifier_list[j])
            feature_dict[model] = id_dict
        reprint(feature_dict, id_dict, model_list, args.train_size)

def cross_pub_val(args, model):
    vocabulary = None
    if model == LinksFeatures:
        vocabulary = HNVocab(args.links, 5000, 0)
    else:
        vocabulary = HNVocab(args.vocabulary, 5000, 50)
    extracter = model(vocabulary)

    if model == EmbededTitleSentencesFeatures:
        classifier = GaussianNB()
    else:
        classifier = MultinomialNB()

    X, ids = extracter.process(args.training, args.train_size)
    ids = np.array(ids)
    return X, ids, classifier

def make_labels(args, X, ids, classifier):
    labeler = Relabeled2Agree()
    y = labeler.process(args.labels, args.train_size)
    pub_labeler = PublisherLabels()
    publishers = pub_labeler.process(args.labels, args.train_size)
    publishers = np.array(publishers)
    num_pubs = np.amax(publishers)
    class_probs_list = []
    id_list = []
    for pub_num in range (0, num_pubs):
        # print(pub_labeler._label_list[pub_num])
        eval_X = X[publishers==pub_num]
        eval_y = y[publishers==pub_num]
        eval_ids = ids[publishers==pub_num]
        train_X = X[publishers!=pub_num]
        train_y = y[publishers!=pub_num]
        classifier.fit(train_X, train_y)
        class_probs = classifier.predict_proba(eval_X)
        y_pred = [np.argmax(x) for x in class_probs]
        # print('current label',eval_y[0])
        # print('percent different',1-np.sum(y_pred==eval_y)/len(y_pred))
        class_probs_list.append(class_probs)
        id_list.append(eval_ids)
    id_list = [val for sublist in id_list for val in sublist]
    class_probs_list = [val for sublist in class_probs_list for val in sublist]
    id_dict = {}
    for i, id in enumerate(id_list):
        id_dict[id] = class_probs_list[i]
    return(id_dict)

def reprint(feature_dict, id_dict, model_list, train_size):
    # train_size is confusing and irrelevant here...
    fp = open('/data/semeval/v4/training/ground-truth-training-bypublisher-20181122.xml','rb')
    out = open('/scratch/kkakkar1/relabelled/iterative_relabels.xml','wb')
    articles = do_xml_parse(fp, 'article', train_size)

    out.write("<?xml version='1.0' encoding='UTF-8' standalone='no'?>".encode())
    out.write("<articles>".encode())
    for a in articles:
        id = a.get("id")
        if id in id_dict:
            print(str(feature_dict[EmbededTitleSentencesFeatures][id]))
            a.set('Links',str(feature_dict[LinksFeatures][id][1]))
            a.set('BagOfWords',str(feature_dict[BagOfWordsFeatures][id][1]))
            a.set('Title',str(feature_dict[EmbededTitleSentencesFeatures][id][0]))
        out.write(etree.tostring(a, pretty_print=True))
    out.write('</articles>'.encode())
    out.close()
