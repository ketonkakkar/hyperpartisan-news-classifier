from abc import ABC, abstractmethod
from itertools import islice
from html import unescape
from scipy import sparse
from lxml import etree
import sys
import argparse
import sys
import numpy as np
from HyperpartisanNewsReader import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB


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


fp = open('/data/semeval/v4/training/ground-truth-training-byarticle-20181122.xml','rb')
out = open('test.xml','wb')

articles = do_xml_parse(fp, 'article')
extracter = Links(vocabulary)
X, ids = extracter.process(args.training, args.train_size)

# for a in articles:
#
#     print(a.set('hyperpartisan', 'orange'))
#     out.write(etree.tostring(a, pretty_print=True))

out.close()
