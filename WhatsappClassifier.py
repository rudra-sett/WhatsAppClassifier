import os
import sys
import numpy as np
import matplotlib as plt
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
import keras
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds

#path to whatsapp chat
filepath = 'domE.txt'

with open(filepath) as fp:
   lines = fp.readlines()

#remove unlabeled lines (a single message can often get split into multiple lines)
#for whatever reason, it doesn't catch all offending lines on one attempt, so it will run 20 times.
for i in range(20):
    for id, line in enumerate(lines):
        print(str(id))
        if (len(line)>0):
            if (line[0] != "8" and line[0] != "7" and line[0] != "6" and line[0] != "5" and line[0] != "4"):
                lines.remove(line)
                print(line)
        if (len(line)==0):
            lines.remove(line)

#strip lines of whitespace
for id, line in enumerate(lines):
    if (line[-1] == '\n'):
        lines[id] = line.rstrip()

#gather all messages into a list, without names
messages = []
for cnt, line in enumerate(lines):
    #print(line)
    if (len(line.split(": "))>1):
        txt = line.split(": ")[1]
        messages.append(txt)

#gather, for each message, the speaker
speakers = []
for cnt, line in enumerate(lines):
    #print(line)
    if (len(line.split(": "))>1):
        print(str(cnt))
        dateperson = line.split(": ")[0]
        person = dateperson.split(" - ")[1]
        speakers.append(person)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(messages)
X = tokenizer.texts_to_sequences(messages)
X = pad_sequences(X)



#get a list of every unique speaker
speakerset = set(speakers)
uspeakers = []
for i in speakerset:
    uspeakers.append(i)

#convert the categories into numbers (index in uspeakers)
Y = np.zeros(shape=(len(speakers),))
for cnt, person in enumerate(speakers):
    Y[cnt] = uspeakers.index(person)

#build the model
embed_dim = 256
lstm_out = 64
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(uspeakers),activation='softmax'))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# Split dataset
train_size = int(0.6 * len(X))
train_X = X[:train_size]
test_X = X[train_size:]
train_Y = Y[:train_size]
test_Y = Y[train_size:]

#train the model
batch_size = 16
model.fit(train_X, train_Y, epochs = 7, batch_size=batch_size, verbose = 2)


#validation
validation_size = int(0.3 * len(X))
X_validate = test_X[-validation_size:]
Y_validate = test_Y[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(test_X, test_Y, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

#this function runs predictions on new material
def guess(txt):
    #vectorizing the message by the pre-fitted tokenizer instance
    txt = [txt]
    txt = tokenizer.texts_to_sequences(txt)
    #padding the tweet to have exactly the same shape as `embedding` input
    txt = pad_sequences(txt, maxlen=X.shape[1], dtype='int32', value=0)
    result = model.predict(txt,batch_size=1,verbose = 2)[0]
    #print("Confidence: " + str(sentiment[np.argmax(sentiment)]*100) + "%")
    print(uspeakers[np.argmax(result)] + "\nConfidence: "+str(result[np.argmax(result)]*100) + "%")

guess("this is a message")
