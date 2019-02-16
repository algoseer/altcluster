import sys
import numpy as np
from tqdm import tqdm

#Load the data

'''
labs=[]
feats=[]

N=len(open(sys.argv[1]).readlines())

with open(sys.argv[1]) as fin:
	for line in tqdm(fin,total=N):
		line=line.rstrip().split(',')
		labs.append(line[0])
		line=np.array(line[1:]).astype('float')
		feats.append(np.reshape(line,(28,28,1))/255.)

feats=np.array(feats)
lablist, labs = np.unique(labs, return_inverse=True)
nb_classes = labs.max()+1
labs = np.eye(nb_classes)[labs]


print(feats.shape, labs.shape, lablist)
N = feats.shape[0]
idx=range(N)

from random import sample
idx=sample(idx,(N-1))
feats=feats[idx]
labs=labs[idx]

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Reshape

model = Sequential()

inp_shape = (28,28,1)
## Convolutional model
model.add(Conv2D(64,3,padding='same',activation='relu',input_shape=inp_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(GlobalAveragePooling2D())

model.add(Dense(nb_classes, activation='softmax'))

model.add(Dense(49*128, activation='relu'))
model.add(Reshape((7,7,128)))
model.add(UpSampling2D(2))
model.add(Conv2D(64,3,padding='same',activation='relu'))
model.add(UpSampling2D(2))
model.add(Conv2D(1,3,padding='same',activation='relu'))


#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#model.fit(feats, labs, epochs=20, batch_size=32, validation_split=0.2)

model.compile(loss='mse',  optimizer='adam')
model.fit(feats, feats, epochs=20, batch_size=32, validation_split=0.2,verbose=True)
mse= model.evaluate(feats,feats)
print()
print("mse=",mse)


from keras.models import Model
bneck_model = Model(inputs=[model.input], outputs=[model.layers[5].output])
bneck_labs = bneck_model.predict(feats)
bneck_hard_labs=np.argmax(bneck_labs,axis=1)
hard_labs = np.argmax(labs, axis=1)

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import mutual_info_score as mi

print("nmi=",nmi(hard_labs, bneck_hard_labs))
print("mi=",mi(hard_labs, bneck_hard_labs))

'''
#TODO Do a tsne on the layer before softmax?
from sklearn.manifold import TSNE
import random
tsne = TSNE(n_components=2, random_state=0)

bneck_feats=Model(model.input,model.layers[4].output).predict(feats[::20])
bneck_tsne = tsne.fit_transform(bneck_feats)
labs_sub = labs[::20]

import matplotlib
matplotlib.use('Agg')
from pylab import *

scatter(bneck_tsne[:,0],bneck_tsne[:,1], color=cm.rainbow(labs_sub.argmax(axis=1)/10.))
for i in range(bneck_tsne.shape[0]):
	if random()<0.2:
		text(bneck_tsne[i,0],bneck_tsne[i,1],str(np.argmax(labs_sub[i])))

savefig('/tmp/fig.png')
clf()
