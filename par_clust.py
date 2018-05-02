import sys
import numpy as np

#Load the data


labs=[]
feats=[]

for line in file(sys.argv[1]):
    line=line.rstrip().split(',')

    labs.append(line[-1])
    feats.append(map(float,line[:-1]))

feats=np.array(feats)
lablist, labs = np.unique(labs, return_inverse=True)
nb_classes = labs.max()+1
labs = np.eye(nb_classes)[labs]


print feats.shape, labs.shape, lablist
N,dim= feats.shape
idx=range(N)

from random import shuffle
shuffle(idx)
feats=feats[idx]
labs=labs[idx]

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
import keras.backend as K

x = Input(shape=(dim,))

M1 = Sequential(name='clf1')
M1.add(Dense(32,activation='relu', input_shape = (dim,)))
M1.add(Dense(16,activation='relu'))
M1.add(Dense(3,activation='softmax'))


M2 = Sequential(name='clf2')
M2.add(Dense(32,activation='relu', input_shape = (dim,)))
M2.add(Dense(16,activation='relu'))
M2.add(Dense(3,activation='softmax'))

M3 = Sequential(name='clf3')
M3.add(Dense(32,activation='relu', input_shape = (dim,)))
M3.add(Dense(16,activation='relu'))
M3.add(Dense(3,activation='softmax'))

y1= M1(x)
y2= M2(x)
y3= M3(x)

xent_loss12=K.categorical_crossentropy(y1,y2)
xent_loss23=K.categorical_crossentropy(y2,y3)
xent_loss31=K.categorical_crossentropy(y3,y1)

#Train the networks to minimize cross entropy between these two branches
model=Model(inputs=[x], outputs=[y1,y2,y3])
model.add_loss(xent_loss12)
model.add_loss(xent_loss23)
model.add_loss(xent_loss31)
model.compile(optimizer='adam')

#model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
#model.fit(feats,[labs,labs], epochs=100, batch_size=64)
model.fit(feats, epochs=50, batch_size=10, validation_split=0.2)

