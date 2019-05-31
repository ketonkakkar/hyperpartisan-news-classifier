##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

from abc import ABC, abstractmethod
from itertools import islice
from html import unescape
from scipy import sparse
from scipy.sparse import csr_matrix
from lxml import etree
from collections import Counter
from nltk.tokenize import word_tokenize
import sys
import numpy as np
import re
import pandas as pd
from urllib.parse import urlparse
from embeding import embed


#####################################################################
# HELPER FUNCTIONS
#####################################################################

def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """
    Parses cleaned up spacy-processed XML files
    """
    fp.seek(0)

    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
    for i, (event, elem) in elements:
        yield elem
        elem.clear()
        if progress_message and (i % 1000 == 0):
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)

def dumb_xml_parse(fp, tag, max_elements=None):
    """
    Parses cleaned up spacy-processed XML files (but not very well)
    """
    elements = etree.parse(fp).findall(tag)
    N = max_elements if max_elements is not None else len(elements)
    return elements[:N]

#####################################################################
# HNVocab
#####################################################################

class HNVocab(object):
    def __init__(self, vocab_file, vocab_size, num_stop_words):
        """
        Creates HNVocab object which is a dicionary mapping words to indicies.
        """
        start_index = 0 if num_stop_words is None else num_stop_words
        end_index = start_index + vocab_size if vocab_size is not None else None

        self._stop = [w.strip() for w in islice(vocab_file, 0, start_index)]
        self._words = [w.strip() for w in islice(vocab_file, start_index, end_index)]
        self._dict = dict([(w, i) for (i, w) in enumerate(self._words)])

    def __len__(self):
        return len(self._dict)

    def index_to_label(self, i):
        return self._words[i]

    def __getitem__(self, key):
        if key in self._dict: return self._dict[key]
        else: return None

#####################################################################
# HNLabels
#####################################################################

class HNLabels(ABC):
    def __init__(self):
        self.labels = None
        self._label_list = None


    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    def process(self, label_file, max_instances=None):
        """
        If the label dictionary is None, it creates a label dictionary which
        maps x to y.
        It then returns the label y for x.
        """
        articles = do_xml_parse(label_file, 'article', max_elements=max_instances)
        y_labeled = list(map(self._extract_label, articles))
        # print("Y labeled", y_labeled)
        if self.labels is None:
            self._label_list = sorted(set(y_labeled))
            # print("list", self._label_list)
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)])
            # print("labels", self.labels)
        y = np.array([self.labels[x] for x in y_labeled])

        return y

    @abstractmethod
    def _extract_label(self, article):
        """ Return the label for this article """
        return "Unknown"

#####################################################################
# HNFeatures
#####################################################################

class HNFeatures(ABC):
    def __init__(self, vocab=None):
        self.vocab = vocab

    def extract_text(self, article):
        """
        Returns the text for a specific article.
        """
        if article.find("spacy") is not None:
            text = unescape("".join([x for x in article.find("spacy").itertext()]).lower()).split()
        else:
            text = []
        return text

    def extract_sentences(self, article):
        """
        Returns the text for a specific article.
        """
        if article.find("spacy") is not None:
            text = unescape("".join([x for x in article.find("spacy").itertext()])).split("-EOS-")
        else:
            text = []
        return text

    def extract_links(self, article):
        """
        Returns the text for a specific article.
        """
        if article.findall('a') is not None:
            link_set = [urlparse(link.items()[0][1]).netloc for link in article.findall('a')]
        else:
            link_set = []
        return link_set

    def extract_tags(self, article):
        """
        Returns the text for a specific article.
        """
        return unescape("".join([x for x in article.find("tag").itertext()]).lower()).split()

    def get_total_links(self, data_file, max_instances=None):
        return None

    def write_links(self, fp, counter):
        return None

    def process(self, data_file, max_instances=None):
        """
        Returns the features for all the examples up to max_instances, and the
        associated ids.
        """
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances
        X = sparse.lil_matrix((N, self._get_num_features()), dtype='uint8')

        ids = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        for i, article in enumerate(articles):
            ids.append(article.get("id"))
            for j, value in self._extract_features(article):
                X[i,j] = value
        return X, ids

    @abstractmethod
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    @abstractmethod
    def _extract_features(self, article):
        """ Returns a list of the features in the article """
        return []

    @abstractmethod
    def _get_num_features(self):
        """ Return the total number of features """
        return -1

#####################################################################

class BinaryLabels(HNLabels):
    def _extract_label(self, article):
        return article.get("hyperpartisan")
class DecisionBased(HNLabels):
    def _extract_label(self, article):
        links = article.get("Links")
        if links == None:
            return article.get("hyperpartisan")
        links = float(links)
        bow = float(article.get("BagOfWords"))
        t = .8
        s = 1 - t
        if links > t:
            return "true"
        elif links < s:
            return "false"
        elif (bow > t) and (links > .56):
            return "true"
        elif (bow <s) and (links < .5):
            return "false"
        else:
            return article.get("hyperpartisan")

class Relabeled2Agree(HNLabels):
    def _extract_label(self, article):
        links = article.get("Links")
        if links == None:
            return article.get("hyperpartisan")
        links = float(links)
        bow = float(article.get("BagOfWords"))
        title = float(article.get("Title"))
        t = .8
        s = 1 - t
        if (links>t and bow>t) or (links>t and title>t) or (bow>t and title>t):
            return "true"
        elif (links<s and bow<s) or (links<s and title<s) or (bow<s and title<s):
            return "false"
        else:
            return article.get("hyperpartisan")

class RelabledMultiply(HNLabels):
    def _extract_label(self, article):
        links = article.get("Links")
        if links == None:
            return  article.get("hyperpartisan")
        bow = float(article.get("BagOfWords"))
        title = float(article.get("Title"))
        prob = links * bow * title
        if prob > .8**3:
            return "true"
        elif prob < .2**3:
            return "false"
        else:
            return article.get("hyperpartisan")


class RelabledAdd(HNLabels):
    def _extract_label(self, article):
        links = article.get("Links")
        if links == None:
            return  article.get("hyperpartisan")
        bow = float(article.get("BagOfWords"))
        title = float(article.get("Title"))
        prob = (links + bow + title)/3
        if prob > .8:
            return "true"
        elif prob < .2:
            return "false"
        else:
            return article.get("hyperpartisan")

class PublisherLabels(HNLabels):
    def _extract_label(self, article):
        return urlparse(article.get("url")).netloc

class BagOfWordsFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    def _extract_features(self, article):
        text = self.extract_text(article)
        count = Counter(text)
        invocab = [(self.vocab._dict[x], count[x]) for x in count if x in self.vocab._dict]
        return invocab

    def _get_num_features(self):
        """ Return the total number of features """
        return len(self.vocab)

class EmbededTitleSentencesFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    def _extract_features(self, article):
        return [article.get('title')]

    def process(self, data_file, max_instances=None):
        """
        Returns the features for all the examples up to max_instances, and the
        associated ids.
        """
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances
        titles = []
        ids = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        for article in articles:
            ids.append(article.get("id"))
            for value in self._extract_features(article):
                titles.append(value)
        X = embed(titles)
        return X, ids

    def _get_num_features(self):
        """ Return the total number of features """
        return 128

class EmbededTitleWordsFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    def _extract_features(self, article):
        return article.get('title').split(' ')

    def process(self, data_file, max_instances=None):
        """
        Returns the features for all the examples up to max_instances, and the
        associated ids.
        """
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances
        titles = []
        ids = []
        X = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        for article in articles:
            ids.append(article.get("id"))
            words = self.extract_text(article)
            if len(words) < 32:
                words += [""]*32
            words = words[:32]
            X.extend(words)
        X = embed(X)
        X = np.array(X).reshape([N,32,128])
        return X, ids

    def _get_num_features(self):
        """ Return the total number of features """
        return 128

class EmbededTextWordsFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    def _extract_features(self, article):
        return self.extract_text(article)

    def process(self, data_file, max_instances=None):
        """
        Returns the features for all the examples up to max_instances, and the
        associated ids.
        """
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances
        titles = []
        ids = []
        X = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        for article in articles:
            ids.append(article.get("id"))
            words = self._extract_features(article)
            if len(words) < 128:
                words += [""]*128
            words = words[:128]
            X.extend(words)
        X = embed(X)
        X = np.array(X).reshape([N,128,128])
        return X, ids

    def _get_num_features(self):
        """ Return the total number of features """
        return 128

class EmbededTextSentencesFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    def _extract_features(self, article):
        sentences = self.extract_sentences(article)
        return sentences

    def process(self, data_file, max_instances=None):
        """
        Returns the features for all the examples up to max_instances, and the
        associated ids.
        """
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances
        titles = []
        ids = []
        X = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        for article in articles:
            ids.append(article.get("id"))
            sentences = self._extract_features(article)
            if len(sentences) < 32:
                sentences += [""]*32
            sentences = sentences[:32]
            X.extend(sentences)
        X = embed(X)
        X = np.array(X).reshape([N,32,128])
        return X, ids

    def _get_num_features(self):
        """ Return the total number of features """
        return 128

class TitleTextFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    def _extract_features(self, article):
        return article.get('title')

    def process(self, data_file, max_instances=None):
        """
        Returns the features for all the examples up to max_instances, and the
        associated ids.
        """
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances
        result = {}
        ids = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        i = 0
        for article in articles:
            ids.append(article.get("id"))
            result[i] = article.get('title')
            i += 1
        X = pd.DataFrame.from_dict(result,orient='index',columns=["sentence"])
        return X, ids

    def _get_num_features(self):
        """ Return the total number of features """
        return None

class LinksFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    def get_total_links(self, data_file, max_instances=None):
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances
        link_set = []
        articles = do_xml_parse(data_file, 'article',
            max_elements=N, progress_message="Article {}")
        for i, article in enumerate(articles):
            links = article.findall('a')
            link_set += [urlparse(link.items()[0][1]).netloc for link in links]
        c = Counter(link_set)
        writeLinks("links.txt", c)

    def write_links(self, fp, counter):
        with open(fp, "w") as f:
            for link in list(counter):
                link += "\n"
                f.write(link)

    def _extract_features(self, article):
        text = self.extract_links(article)
        count = Counter(text)
        # invocab = [(self.vocab._dict[x], count[x]) for x in count if x in self.vocab._dict]
        invocab = [(self.vocab._dict[x], 1) for x in count if x in self.vocab._dict and count[x] > 0]
        return invocab

    def _get_num_features(self):
        """ Return the total number of features """
        return len(self.vocab)
