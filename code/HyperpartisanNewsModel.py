import numpy as np
from utilities import *
from os import mkdir
from HyperpartisanNewsReader import *
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from embeding import embed
import sys

#####################################################################
# HNModel
#####################################################################

class HNModel(ABC):
    def __init__(self,vocabulary=None,clf=None):
        self.labeler = None
        self.extracter = None
        self.clf = clf

    def train(self,label_file,train_file,train_size,xvalidate=None):
        y_old = BinaryLabels().process(label_file,train_size)
        y = self.labeler.process(label_file,train_size)
        print("relabeled:",np.sum(y_old!=y)/len(y))
        publisher_labeler = PublisherLabels()
        publishers = publisher_labeler.process(label_file,train_size)
        X, ids = self.extracter.process(train_file,train_size)
        print(np.shape(y))
        self.clf.fit(X, y)
        y_pred = self.clf.predict(X)
        print("training data metrics:")
        print_metrics(y,y_pred)
        if xvalidate != None:
            train_index, test_index  = next(KFold(n_splits=xvalidate,shuffle=True, random_state=1).split(X))
            self.clf.fit(X[train_index], y[train_index])
            y_pred = self.clf.predict(X[test_index])
            print_metrics(y[test_index],y_pred)
            for name, i in publisher_labeler.labels.items():
                print(name)
                print_metrics(y[test_index][publishers[test_index]==i], y_pred[publishers[test_index]==i])
    def test(self,label_file,test_file,test_size):
        y_test = self.labeler.process(label_file,test_size)
        publisher_labeler = PublisherLabels()
        publishers = publisher_labeler.process(label_file,test_size)
        X_test, ids = self.extracter.process(test_file,test_size)
        y_pred = self.clf.predict(X_test)
        print("test data metrics:")
        print_metrics(y_test,y_pred)
        for name, i in publisher_labeler.labels.items():
            print(name)
            print_metrics(y_test[publishers==i],y_pred[publishers==i])

    def predict(self,test_file,test_size):
        X_test, ids = self.extracter.process(test_file,test_size)
        y_pred = self.clf.predict(X_test)
        y_pred = np.expand_dims(y_pred, axis=1)
        return y_pred,ids

    def predict_proba(self,test_file,test_size):
        X_test, ids = self.extracter.process(test_file,test_size)
        y_probs = self.clf.predict_proba(X_test)[:,1]
        y_probs = np.expand_dims(y_probs, axis=1)
        return y_probs,ids

    def get_features(self,train_file,train_size):
        return self.extracter.process(train_file,train_size)

    def save(self,save_name):
        try:
            mkdir('saved_models/'+save_name)
            print(save_name ,  "saved")
        except FileExistsError:
            print(save_name ,  "updated")
        joblib.dump(self.clf, 'saved_models/'+save_name+'/clf.joblib')
        joblib.dump(self.extracter, 'saved_models/'+save_name+'/extracter.joblib')

    def restore(self,save_name):
        self.clf = joblib.load('saved_models/'+save_name+'/clf.joblib')
        self.extracter = joblib.load('saved_models/'+save_name+'/extracter.joblib')

#####################################################################

class TitleMeaning(HNModel):
    def __init__(self,vocabulary=None,clf=GaussianNB()):
        self.labeler = BinaryLabels()
        self.extracter = EmbededTitleFeatures()
        self.clf = clf

class BagOfWords(HNModel):
    def __init__(self,vocabulary=None,clf=MultinomialNB()):
        self.labeler = BinaryLabels()
        self.extracter = BagOfWordsFeatures(vocabulary)
        self.clf = clf

class Links(HNModel):
    def __init__(self,vocabulary=None,clf=MultinomialNB()):
        self.labeler =  Relabeled2Agree()
        self.extracter = LinksFeatures(vocabulary)
        self.clf = clf
