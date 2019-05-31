import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def make_embed_fn(module):
  with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    embed_fn = hub.Module(module)
    embeddings = embed_fn(sentences)
    session = tf.train.MonitoredSession()
  return lambda x: session.run(embeddings, {sentences: x})

def embed(sentences):
    embed_fn = make_embed_fn("https://tfhub.dev/google/nnlm-en-dim128/1")
    #embed_fn = make_embed_fn("https://tfhub.dev/google/universal-sentence-encoder/2")
    for i in range(0,len(sentences),100000):
        print('Embeded sentences:',str(i), end='\r')
        if i == 0:
            X = embed_fn(sentences[i:i+100000])
        else:
            X = np.concatenate((X,embed_fn(sentences[i:i+100000 % len(sentences)])))
    return X
