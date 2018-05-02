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

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_shape=(dim,),activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(dim))

#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#model.fit(feats, labs, epochs=20, batch_size=32, validation_split=0.2)

model.compile(loss='mse',  optimizer='adam')
model.fit(feats, feats, epochs=200, batch_size=32, validation_split=0.2,verbose=False)
mse= model.evaluate(feats,feats)
print
print "mse=",mse

from keras.models import Model
bneck_model = Model(inputs=[model.input], outputs=[model.layers[2].output])
bneck_labs = bneck_model.predict(feats)
bneck_hard_labs=np.argmax(bneck_labs,axis=1)
hard_labs = np.argmax(labs, axis=1)

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import mutual_info_score as mi

print "nmi=",nmi(hard_labs, bneck_hard_labs)
print "mi=",mi(hard_labs, bneck_hard_labs)

#Compute probabilistic mutual information
J= np.dot(labs.T, bneck_labs)/N
K= np.outer(J.sum(axis=0), J.sum(axis=1))
H1= bneck_labs*np.log(bneck_labs+1e-16)
H1 = -H1.sum()/N
H2= labs*np.log(labs+1e-16)
H2= -H2.sum()/N

pmi = J*np.log(J/K)
print "Prob MI=",pmi.sum()
print H1, H2
