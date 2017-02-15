import numpy as np
import random

#Function for converting label vector to one hot vector
def toOneHot(label):
	label = label.astype(np.int)
	label_length=label.size
	label_unique=np.unique(label)
	oneHot=np.zeros([label_length,label_unique.size])
	for i in range(label_length):
		oneHot[i,label[i]-1]=1
	return oneHot
  
  
#Read text files and convert them in numpy arrays
f=open('AlexNetWithRoom.fvec','r')
fvec=np.reshape(np.fromstring(f.readline(),sep=','),[-1,1])
for line in f:
	linevec=np.reshape(np.fromstring(line,sep=','),[-1,1])
	fvec = np.append(fvec,linevec,axis=1)
fvec=fvec.T
f.close

f2=open('AlexNetWithRoom.label','r')
label=np.reshape(np.fromstring(f2.read(),sep='\n'),[-1,1])
f2.close

label_oneHot = toOneHot(label)
f3=open('AlexNetWithRoom.room','r')
room=np.reshape(np.fromstring(f3.read(),sep='\n'),[-1,1])
f3.close

#Shuffle the vectros and labels
idx_shuffle = range(1332)
random.shuffle(idx_shuffle)
fvec_shuffle= fvec[idx_shuffle,:]
label_oneHot_shuffle=label_oneHot[idx_shuffle,:]
label_shuffle=label[idx_shuffle,:]
room_shuffle=room[idx_shuffle]
