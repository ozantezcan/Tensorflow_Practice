import numpy as np
import matplotlib.pyplot as plt

def cifarImShow(imgs,label,number):
  img = imgs[number,:,:,:]
  imgplot=plt.imshow(img)
  plt.title('Class'+str(label[number]))
  plt.show()

def toOneHot(label):
  label_length=label.size
  label_unique=np.unique(label)
  oneHot=np.zeros([label_length,label_unique.size])
  for i in range(label_length):
    oneHot[i,label[i]]=1
  return oneHot
  
train_data=np.load('data_batch_1')
train_img = train_data['data'].reshape(10000,3,32,32).transpose(0,2,3,1)
train_label = np.asarray(train_data['labels'])
for i in range(2,6):
  current_data = np.load('data_batch_'+str(i))
  current_img = current_data['data'].reshape(10000,3,32,32).transpose(0,2,3,1)
  train_img = np.append(train_img,current_img,axis=0)
  train_label = np.append(train_label,np.asarray(current_data['labels']))
train_label_oneHot=toOneHot(train_label)

test_data=np.load('test_batch')
test_img = train_data['data'].reshape(10000,3,32,32).transpose(0,2,3,1)
test_label = np.asarray(train_data['labels'])
test_label_oneHot=toOneHot(test_label)

train_img=train_img.astype(np.float32)/255
test_img=test_img.astype(np.float32)/255


