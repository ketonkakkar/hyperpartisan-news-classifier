import keras.backend as K
import keras.layers
from keras import optimizers
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Activation, Lambda, Dropout
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
from typing import List, Tuple

def compiled_dnn(num_feat,  # type: int
                 shape,  # type: int
                 dropout_rate=0.05,  # type: float
                 ):

    model = Sequential()
    model.add(Dense(shape[0], input_dim=num_feat, activation='relu'))
    for i in range(1,len(shape)):
        model.add(Dropout(dropout_rate))
        model.add(Dense(shape[i], input_dim=shape[i-1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
