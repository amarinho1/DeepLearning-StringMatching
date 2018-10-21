#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import csv
import time
import unicodedata
import numpy as np
from layers import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Flatten, Masking, Dense, Input, Dropout, LSTM, GRU, Bidirectional, Layer, Masking, Lambda, Permute, Highway, TimeDistributed
from keras.layers.merge import concatenate, multiply, subtract
from keras import backend as K
from keras import initializers
from theano.tensor import _shared


def deep_neural_net_gru(train_data_1, train_data_2, train_labels, test_data_1, test_data_2, test_labels, max_len,
                        len_chars,  bidirectional, hidden_units, selfattention , maxpooling, alignment , shortcut , multiplerlu , onlyconcat, n):
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    checkpointer = ModelCheckpoint(filepath="/home/amarinho/data-amarinho/checkpoint" + str(n) +".hdf5", verbose=1, save_best_only=True)
    lstm1 = GRU(hidden_units, implementation=2, return_sequences=True, name='lstm1' )
    lstm2 = GRU(hidden_units, implementation=2, return_sequences=True, name='lstm2')
    lstm3 = GRU(hidden_units, implementation=2, return_sequences=True, name='lstm3') 
    lstm1 = Bidirectional(lstm1, name='bilstm1')
    lstm2 = Bidirectional(lstm2, name='bilstm2')
    lstm3 = Bidirectional(lstm3, name='bilstm3')

    input_word1 = Input(shape=(max_len, len_chars))
    input_word2 = Input(shape=(max_len, len_chars))

    mask = Masking(mask_value=0, input_shape=(max_len, len_chars))(input_word1)
    l1 = lstm1(mask)
    l1 = Dropout(0.01)(l1)
    l1 = MaxPooling1DMasked(pool_size=1, name = 'maxpooling' )(l1)

    input_concat = concatenate([l1, mask])
    l2 = lstm2(input_concat)
    l2 = Dropout(0.01)(l2)
    l2 = MaxPooling1DMasked(pool_size=1, name = 'maxpooling2' )(l2)

    input_concat = concatenate([mask, l2])
    l3 = lstm3(input_concat)
    l3 = Dropout(0.01)(l3)
    l3 = MaxPooling1DMasked(pool_size=1, name = 'maxpooling3' )(l3)

    final_input_concat = concatenate([l1, l2, l3], axis=1)
    SentenceEncoder = Model(input_word1, final_input_concat)

    word1_representation = SentenceEncoder(input_word1)
    word2_representation = SentenceEncoder(input_word2)

    concat = concatenate([word1_representation, word2_representation])
    mul = multiply([word1_representation, word2_representation])
    sub = subtract([word1_representation, word2_representation])

    final_merge = concatenate([concat, mul, sub])
    dropout3 = Dropout(0.01)(final_merge)
    dense1 = Dense(hidden_units*2, activation='relu', name='dense1')(dropout3)
    dropout4 = Dropout(0.01)(dense1)
    flatten = Flatten()(dense1)
    dropout5 = Dropout(0.01)(flatten)
    dense2 = Dense(1, activation='sigmoid', name='dense2')(dropout5)
    final_model = Model([input_word1, input_word2], dense2)
    print(final_model.summary())

    print('Compiling...')
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('Fitting...')
    final_model.fit([train_data_1, train_data_2], train_labels, verbose = 0, validation_data=([test_data_1, test_data_2], test_labels), 
		    callbacks=[checkpointer, early_stop], epochs=20)

    start_time = time.time()
    aux1 = final_model.predict([test_data_1, test_data_2], verbose = 0)
    aux = (aux1 > 0.5).astype('int32').ravel()
    return aux, (time.time() - start_time)

def evaluate_deep_neural_net(dataset='dataset-string-similarity.txt', method='gru', training_instances=-1,
                             bidirectional=True, hiddenunits=60, selfattention=False , maxpooling=False , 
                        alignment = False , shortcut=True , multiplerlu=False , onlyconcat=False):
    max_seq_len = 40
    num_true = 0.0
    num_false = 0.0
    num_true_predicted_true = 0.0
    num_true_predicted_false = 0.0
    num_false_predicted_true = 0.0
    num_false_predicted_false = 0.0
    timer = 0.0
    with open(dataset) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res"], delimiter='|')
        for row in reader:
            if row['res'] == "1":
                num_true += 1.0
            else:
                num_false += 1.0
    print(num_false);
    XA1 = []
    XB1 = []
    XC1 = []
    Y1 = []
    XA2 = []
    XB2 = []
    XC2 = []
    Y2 = []
    start_time = time.time()
    print( "Reading dataset... " + str(start_time - start_time))
    with open(dataset) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "id1", "id2", "lat1", "lon1", "lat2", "lon2", "s3"], delimiter='|')
        start_time = time.time()
        for row in reader:
            if row['res'] == "1":
                if len(Y1) < ((num_true + num_false) / 2.0):
                    Y1.append(1)
                else:
                    Y2.append(1)
            else:
                if len(Y1) < ((num_true + num_false) / 2.0):
                    Y1.append(0)
                else:
                    Y2.append(0)
            row['s1'] = row['s1'] #.decode('utf-8')
            row['s2'] = row['s2'] #.decode('utf-8')
            row['s1'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s1'] + u'|')), encoding='utf-8')
            row['s2'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s2'] + u'|')), encoding='utf-8')
            if len(XA1) < ((num_true + num_false) / 2.0):
                XA1.append(row['s1'])
                XB1.append(row['s2'])
            else:
                XA2.append(row['s1'])
                XB2.append(row['s2'])
        print( "Dataset read... " + str(time.time() - start_time))
    Y1 = np.array(Y1, dtype=np.bool)
    Y2 = np.array(Y2, dtype=np.bool)
    chars = list(set(list([val for sublist in XA1 + XB1 + XC1 + XA2 + XB2 + XC2 for val in sublist])))
    char_labels = {ch: i for i, ch in enumerate(chars)}
    aux1 = np.memmap("/home/amarinho/data-amarinho/temporary-file-dnn-1-" + method, mode="w+", shape=(len(XA1), max_seq_len, len(chars)), dtype=np.bool)
    for i, example in enumerate(XA1):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XA1 = aux1
    aux1 = np.memmap("/home/amarinho/data-amarinho/temporary-file-dnn-2-" + method, mode="w+", shape=(len(XB1), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XB1):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XB1 = aux1
    aux1 = np.memmap("/home/amarinho/data-amarinho/temporary-file-dnn-3-" + method, mode="w+", shape=(len(XA2), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XA2):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XA2 = aux1
    aux1 = np.memmap("/home/amarinho/data-amarinho/temporary-file-dnn-4-" + method, mode="w+", shape=(len(XB2), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XB2):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XB2 = aux1
    print ("Temporary files created... " + str(time.time() - start_time))
    print ("Training classifiers...")

    if training_instances <= 0: training_instances = min(len(Y1), len(Y2))

    aux1, time1 = deep_neural_net_gru(train_data_1=XA1[0:training_instances, :, :],
                                      train_data_2=XB1[0:training_instances, :, :],
                                      train_labels=Y1[0:training_instances, ], test_data_1=XA2, test_data_2=XB2,
                                      test_labels=Y2, max_len=max_seq_len, len_chars=len(chars),
                                      bidirectional=bidirectional, hidden_units=hiddenunits, selfattention=selfattention , maxpooling=maxpooling , 
                                      alignment = alignment , shortcut=shortcut , multiplerlu=multiplerlu , onlyconcat=onlyconcat,n=1)

    aux2, time2 = deep_neural_net_gru(train_data_1=XA2[0:training_instances, :, :],
                                      train_data_2=XB2[0:training_instances, :, :],
                                      train_labels=Y2[0:training_instances, ], test_data_1=XA1, test_data_2=XB1,
                                      test_labels=Y1, max_len=max_seq_len, len_chars=len(chars),
                                      bidirectional=bidirectional, hidden_units=hiddenunits, selfattention=selfattention , maxpooling=maxpooling , 
                                      alignment = alignment , shortcut=shortcut , multiplerlu=multiplerlu , onlyconcat=onlyconcat,n=2)
    timer = time1 + time2
    print( "Total Time :", timer)
    print( "Matching records...")
    real = list(Y1) + list(Y2)
    #print(real)
    #print(len(real))
    predicted = list(aux2) + list(aux1)
    #print(predicted)
    #print(len(predicted))
    file = open("/home/amarinho/data-amarinho/dataset-dnn-accuracy","w+")
    for pos in range(len(real)):
        if float(real[pos]) == 1.0:
            if float(predicted[pos]) == 1.0:
                num_true_predicted_true += 1.0
                file.write("TRUE\tTRUE\n")
            else:
                num_true_predicted_false += 1.0
                file.write("TRUE\tFALSE\n")
        else:
            if float(predicted[pos]) == 1.0:
                num_false_predicted_true += 1.0
                file.write("FALSE\tTRUE\n")
            else:
                num_false_predicted_false += 1.0
                file.write("FALSE\tFALSE\n")


    timer = (timer / float(int(num_true + num_false))) * 50000.0
    acc = (num_true_predicted_true + num_false_predicted_false) / (num_true + num_false)
    pre = (num_true_predicted_true) / (num_true_predicted_true + num_false_predicted_true)
    rec = (num_true_predicted_true) / (num_true_predicted_true + num_true_predicted_false)
    f1 = 2.0 * ((pre * rec) / (pre + rec))
    file.close()
    print ("Metric = Deep Neural Net Classifier :", method.upper())
    print ("Bidirectional :", bidirectional)
    print ("Highway Network :", multiplerlu)
    print ("Shortcut Connections:", shortcut)
    print ("Maxpolling :", maxpooling)
    print ("Inner Attention :", selfattention)
    print ("Hard Allignment Attention :", alignment)
    print ("Accuracy =", acc)
    print ("Precision =", pre)
    print ("Recall =", rec)
    print ("F1 =", f1)
    print ("Processing time per 50K records =", timer)
    print ("Number of training instances =", training_instances)
    print ("")
    os.remove("/home/amarinho/data-amarinho/temporary-file-dnn-1-" + method)
    os.remove("/home/amarinho/data-amarinho/temporary-file-dnn-2-" + method)
    os.remove("/home/amarinho/data-amarinho/temporary-file-dnn-3-" + method)
    os.remove("/home/amarinho/data-amarinho/temporary-file-dnn-4-" + method)
    sys.stdout.flush()

evaluate_deep_neural_net(dataset=sys.argv[1])
