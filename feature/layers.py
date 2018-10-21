#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import csv
import time
import unicodedata
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Masking, Dense, Input, Dropout, LSTM, GRU, Bidirectional, MaxPooling1D, GlobalMaxPooling1D, Layer, Masking, Lambda, Permute, Highway, TimeDistributed
from keras import backend as K
from theano.tensor import _shared
if K.backend() == 'theano':
    from keras import initializers, regularizers, constraints
else:
    from keras import initializations


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMaxPooling1DMasked, self).__init__(**kwargs)
    def build(self, input_shape): super(GlobalMaxPooling1DMasked, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def call(self, x, mask=None): return super(GlobalMaxPooling1DMasked, self).call(x)

class MaxPooling1DMasked(MaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaxPooling1DMasked, self).__init__(**kwargs)
    def build(self, input_shape): super(MaxPooling1DMasked, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        return None
    def call(self, x, mask=None): return super(MaxPooling1DMasked, self).call(x)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
       
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


class SelfAttLayer(Layer):
    def __init__(self, **kwargs):
        self.attention = None
        self.init = initializations.get('normal')
        self.supports_masking = True
        super(SelfAttLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(SelfAttLayer, self).build(input_shape)
    def call(self, x, mask=None):
        eij = K.tanh(dot_product(x, self.W))
        ai = K.exp(eij)
        weights = ai/K.cast(K.sum(ai, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x*K.expand_dims(weights)
        self.attention = weights
        return K.sum(weighted_input, axis=1)
    def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[-1])



class AlignmentAttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(AlignmentAttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(AlignmentAttentionLayer, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def call(self, inputs, mask=None):
        input1 = inputs[0]
        input2 = inputs[1]

        eij = dot_product(input1, K.transpose(input2))
        eij = K.tanh(eij)
        a = K.exp(eij)
        return a
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        #a /= K.sum(a, axis=1)
        weighted_input = input1 * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)
        print(result)
        

    def compute_output_shape(self, input_shape): return input_shape[0]

def AlignmentAttention(input_1, input_2):
    def unchanged_shape(input_shape): return input_shape
    def softmax(x, axis=-1):
        ndim = K.ndim(x)
        if ndim == 2: return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else: raise ValueError('Cannot apply softmax to a tensor that is 1D')
    w_att_1 = Sequential()
    w_att_1.add(Merge([input_1, input_2], mode='dot', dot_axes=-1))
    w_att_1.add(Lambda(lambda x: softmax(x, axis=1), output_shape=unchanged_shape))
    w_att_2 = Sequential()
    w_att_2.add(Merge([input_1, input_2], mode='dot', dot_axes=-1))
    w_att_2.add(Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape))    
    w_att_2.add(Permute((2,1)))
    in1_aligned = Sequential()
    in1_aligned.add(Merge([w_att_1, input_1], mode='dot', dot_axes=1))    
    in2_aligned = Sequential()
    in2_aligned.add(Merge([w_att_2, input_2], mode='dot', dot_axes=1))
    q1_combined = Sequential()
    q1_combined.add(Merge([input_1,in2_aligned], mode='concat'))
    q2_combined = Sequential()
    q2_combined.add(Merge([input_2,in1_aligned], mode='concat'))
    return q1_combined, q2_combined
