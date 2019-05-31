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
sys.path.insert(0, 'code/kerastcn')
from tcn import compiled_tcn
from dnn import compiled_dnn

#####################################################################
# HNModel
#####################################################################

class HNKerasModel(ABC):
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
        for name, i in publisher_labeler.labels.items():
            print(name)
            print("current label:",y_old[publishers==i][0])
            print("relabeled:",np.sum(y_old[publishers==i]!=y[publishers==i])/len(y[publishers==i]))
        X, ids = self.extracter.process(train_file,train_size)
        if xvalidate == None:
            self.clf.fit(X, y,epochs=10)
        else:
            train_index, test_index  = next(KFold(n_splits=xvalidate,shuffle=True, random_state=1).split(X))
            self.clf.fit(X[train_index], y[train_index], epochs=10,validation_data=(X[test_index], y[test_index]))

    def test(self,label_file,test_file,test_size):
        y_test = self.labeler.process(label_file,test_size)
        publisher_labeler = PublisherLabels()
        publishers = publisher_labeler.process(label_file,test_size)
        X_test, ids = self.extracter.process(test_file,test_size)
        print("test data metrics:")
        print("accuracy:",self.clf.evaluate(X_test,y_test)[1])
        y_pred = np.argmax(self.clf.predict(X_test),axis=1)
        for name, i in publisher_labeler.labels.items():
            print(name)
            print("accuracy:",np.sum(y_test[publishers==i]==y_pred[publishers==i])/len(y_test[publishers==i]))

    def predict(self,test_file,test_size):
        X_test, ids = self.extracter.process(test_file,test_size)
        y_pred = self.clf.predict(X_test)
        if np.shape(y_pred)[1] == 2:
          y_pred = np.argmax(y_pred)
        else:
          y_pred = np.where(y_pred>.5,1,0)
        y_pred = np.expand_dims(y_pred, axis=1)
        return y_pred,ids

    def predict_proba(self,test_file,test_size):
        X_test, ids = self.extracter.process(test_file,test_size)
        y_probs = self.clf.predict(X_test)
        if np.shape(y_probs)[1] == 2:
          y_probs = y_probs[:,1]
          y_probs = np.expand_dims(y_probs, axis=1)
        return y_probs,ids

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

##################################################################################
class TextSentences(HNKerasModel):
    def __init__(self,vocabulary=None,clf=None):
        self.labeler = BinaryLabels()
        #self.labeler = Relabeled2Agree()
        self.extracter = EmbededTextSentencesFeatures()
        self.clf = compiled_tcn(nb_filters=128, kernel_size=2, nb_stacks=1, dilations= [1, 2, 4, 8, 16, 32],
            activation='norm_relu', padding='causal', use_skip_connections=True,
            dropout_rate=0.5, return_sequences=False,num_feat=128,num_classes=2,max_len=32)

class TitleSentences(HNKerasModel):
    def __init__(self,vocabulary=None,clf=None):
        self.labeler = BinaryLabels()
        #self.labeler = Relabeled2Agree()
        self.extracter = EmbededTitleSentencesFeatures()
        self.clf = compiled_dnn(num_feat=128,shape=[256,56],dropout_rate=0.5)

class TitleWords(HNKerasModel):
    def __init__(self,vocabulary=None,clf=None):
        self.labeler = BinaryLabels()
        #self.labeler = Relabeled2Agree()
        self.extracter = EmbededTitleWordsFeatures()
        self.clf = compiled_tcn(nb_filters=128, kernel_size=2, nb_stacks=1, dilations= [1, 2, 4, 8, 16, 32],
            activation='norm_relu', padding='causal', use_skip_connections=True,
            dropout_rate=0.5, return_sequences=False,num_feat=128,num_classes=2,max_len=32)

class TextWords(HNKerasModel):
    def __init__(self,vocabulary=None,clf=None):
        self.labeler = BinaryLabels()
        #self.labeler = Relabeled2Agree()
        self.extracter = EmbededTextWordsFeatures()
        self.clf = compiled_tcn(nb_filters=128, kernel_size=2, nb_stacks=4, dilations= [1, 2, 4, 8, 16, 32,64],
            activation='norm_relu', padding='causal', use_skip_connections=True,
            dropout_rate=0.5, return_sequences=False,num_feat=128,num_classes=2,max_len=128)
