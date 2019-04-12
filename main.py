# import tensorflow as tf 
# import keras
import pandas as pd
import numpy as np
import string
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import LSTM
import pickle

DATA = 'rapKeene.txt'
LOG_DIR = './logDir'


maxLineLen = 0
c=0
C=0

with open(DATA) as f:
    text = f.read()
    C = len(text) 
    print(type(text))
    s = text.encode('ascii',errors='ignore') #clean text
    c = len(str(s)) 

def getBatch(feat,lab):
    l,_,_= feat.shape
    choices = np.random.choice(l, size=BATCH_SIZE)
    return feat[choices],lab[choices]

print('total',C)
print('Small',c)
print(s[9])

chars = sorted(list(set(s)))
mapping = dict((c, i) for i, c in enumerate(chars))
#97 chars 


for i in range(len(s)):
    print(s[i:i+10])




# from keras.preprocessing.text import Tokenizer
# from keras.layers import Embedding, Dense, Flatten, Conv1D, Dropout # Rnn, lstm
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import pad_sequences
# from keras import regularizers

'''
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()


for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)

# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))

'''



'''
import keras
import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, Flatten, Conv1D, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

# Read Data from CSV files header = none to not ignore first row
train = pd.read_csv("./ag_news/train.csv",header=None)
test = pd.read_csv("./ag_news/test.csv",header=None)

#label the collumns
train.columns   = ['cat','title','description']
test.columns    = ['cat','title','description']

train['space'] = " "
train["text"] = train.title + train.space + train.description
test['space'] = " "
test["text"] = test.title + test.space + test.description

# split the data apart for val
train, val = train_test_split(train,test_size=.05)

# Create a Tokenizer instance
tokenizer = Tokenizer() 
# Only  fit on text this for training data
tokenizer.fit_on_texts(train.text)

# Create sequenced data for the vocabs
trainSeq = tokenizer.texts_to_sequences(train.text)
valSeq   = tokenizer.texts_to_sequences(val.text)
testSeq  = tokenizer.texts_to_sequences(test.text)

# Pad the vocabs so that they are all the same len
lenPad=185
trainPad = pad_sequences(trainSeq,maxlen=lenPad)
valPad   = pad_sequences(valSeq,maxlen=lenPad)
testPad  = pad_sequences(testSeq,maxlen=lenPad)

# how long to make the embedding vector
embeddingVectorLen = 32
vocabLength = len(tokenizer.word_index)

# one hot encode the labels 
train_cat = np.array(train.cat-1)
train_cat = train_cat.reshape(train_cat.shape[0],1)
train_cat = keras.utils.to_categorical(train_cat,num_classes=4)

val_cat = np.array(val.cat-1)
val_cat = val_cat.reshape(val_cat.shape[0],1)
val_cat = keras.utils.to_categorical(val_cat,num_classes=4)

test_cat = np.array(test.cat-1)
test_cat = test_cat.reshape(test_cat.shape[0],1)
test_cat = keras.utils.to_categorical(test_cat,num_classes=4)


model = Sequential()
model.add(Embedding(vocabLength+1, embeddingVectorLen, input_length=lenPad))
#model.add(Conv1D(8,32,padding = "same",kernel_regularizer=regularizers.l2()))
#model.add(Conv1D(128,32,padding ="same",dilation_rate=3,kernel_regularizer=regularizers.l2()))

model.add(Dropout(.7))
model.add(Flatten())
model.add(Dense(4,activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.fit(trainPad, train_cat,
            batch_size=64,
                epochs = 2,
                    verbose = 1,
                        validation_data =(valPad,val_cat))

score = model.evaluate(testPad,test_cat,verbose= 1)
print("Test loss:",score[0])
print("Test accuracy:",score[1])
'''
