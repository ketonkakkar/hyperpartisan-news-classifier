import numpy as np
from utilities import *
from os import mkdir
from HyperpartisanNewsReader import *
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from HyperpartisanNewsModel import *
sys.path.insert(0, 'code/kerastcn')
from HyperpartisanNewsKerasModel import *
from embeding import embed

#####################################################################
# HNEnsemble
#####################################################################

class HNEnsemble():
    def __init__(self,models_file,clf=MultinomialNB(),labeler=BinaryLabels()):
        self.models = self._load_models(models_file)
        self.clf = clf
        self.clf2 = Multiply()
        self.clf3 = LinksOrVote()
        self.labeler = labeler

    def _load_models(sekf,models_file):
        model_dict = {'BagOfWords':BagOfWords, 'Links':Links,
            'TextSentences':TextSentences, 'TextWords':TextWords,
            'TitleSentences':TitleSentences, 'TitleWords':TitleWords}
        lines = models_file.readlines()
        models = []
        for line in lines:
            [model,save_name] = line.strip('\n').split()
            model = model_dict[model]()
            model.restore(save_name)
            models.append(model)
        return models

    def train(self,label_file,train_file,train_size):
        y = self.labeler.process(label_file,train_size)
        first = True
        for model in self.models:
            if first:
                X,ids = model.predict_proba(train_file,train_size)
                first = False
            else:
                X = np.append(X,model.predict_proba(train_file,train_size)[0],axis=1)
        self.clf.fit(X, y)
        y_pred = self.clf.predict(X)
        print("training data metrics:")
        print_metrics(y,y_pred)
        print("multiply")
        y_pred = self.clf2.predict(X)
        print_metrics(y,y_pred)
        print("links or vote")
        y_pred = self.clf3.predict(X)
        print_metrics(y,y_pred)

    def test(self,label_file,test_file,test_size):
        y_test = self.labeler.process(label_file,test_size)
        first = True
        for model in self.models:
            if first:
                X_test,ids = model.predict_proba(test_file,test_size)
                first = False
            else:
                X_test = np.append(X_test,model.predict_proba(test_file,test_size)[0],axis=1)
        y_pred = self.clf.predict(X_test)
        print("test data metrics:")
        print_metrics(y_test,y_pred)
        print("multiply")
        y_pred = self.clf2.predict(X_test)
        print_metrics(y_test,y_pred)
        print("links or vote")
        y_pred = self.clf3.predict(X_test)
        print_metrics(y_test,y_pred)

    def predict(self,test_file,test_size):
        first = True
        for model in self.models:
            if first:
                X_test,ids = model.predict(train_file,train_size)
                first = False
            else:
                X_test = np.append(X_test,model.predict(test_file,test_size),axis=1)
        y_pred = self.clf.predict(X_test)
        return y_pred,ids

    def predict_proba(self,test_file,test_size):
        first = True
        for model in self.models:
            if first:
                X_test,ids = model.predict_proba(train_file,train_size)
                first = False
            else:
                X_test = np.append(X_test,model.predict_proba(test_file,test_size),axis=1)
        y_pred = self.clf.predict_proba(X_test)
        return y_pred,ids

    def save(self,save_name):
        try:
            mkdir('saved_ensembles/'+save_name)
            print(save_name ,  "saved")
        except FileExistsError:
            print(save_name ,  "updated")
        joblib.dump(self.clf, 'saved_ensembles/'+save_name+'/clf.joblib')

    def restore(self,save_name):
        self.clf = joblib.load('saved_ensembles/'+save_name+'/clf.joblib')

class Vote:
    def __init__(self):
        None
    def fit(self,X,y):
        None
    def predict(self,X):
        X = np.where(X > 0.5, 1, 0)
        votes = np.sum(X,1)
        return np.where(votes > 1.5, 1, 0)

class Multiply:
    def __init__(self):
        None
    def fit(self,X,y):
        None
    def predict(self,X):
        prod = np.prod(X,1)
        return np.where(prod > .5**3, 1, 0)

class LinksOrVote:
    def __init__(self):
        None
    def fit(self,X,y):
        None
    def predict(self,X):
        new_X = np.where(X > 0.5, 1, 0)
        votes = np.sum(new_X,1)
        voting =  np.where(votes > 1.5, 1, 0)
        links = new_X[:,2]
        results = np.where(links > .8, np.ones_like(voting), voting)
        results = np.where(links < .2, np.zeros_like(results),results)
        return results
