
# coding: utf-8

# In[3]:


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
import sklearn
from sklearn.model_selection import train_test_split
import theano
from theano import config
import os 
from keras.models import load_model
import matplotlib.image as mpimg
from random import shuffle
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")
#os.environ['THEANO_FLAGS'] = "device=gpu" 
theano.config.floatX = 'float32'
#theano.config.device='gpu0'
learning_rate=0.00001


# In[15]:

def gamma(image):
    gamma = np.random.uniform(0.3, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def generator(samples, batch_size=64):
    num_samples = len(samples)
    #print(generator)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    #name = './data/IMG/'+batch_sample[0].split('/')[-1]
                    #print(name)
                    correction=0.25          
                    center_image = mpimg.imread(name)#cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    #center_image=crop_image(center_image)
                    if i==0:
                    #   angles.append(center_angle)
                       center_angle=center_angle
                    if i==1:
                       center_angle+=correction
                       #print('left')
                       #angles.append(center_angle+correction)
                    if i==2:
                       center_angle-=correction
                       #print('right')
                       #angles.append(center_angle-correction) 
                    #print(center_angle)
                    #plt.imshow(center_image)
                    #plt.show()
                    center_image=cv2.resize(center_image[60:140,:],(64,64))
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    #center_image = cv2.GaussianBlur(center_image, (3,3), 0)
                    #recen=gamma(center_image)
                    #plt.imsave('crop_image.png',center_image)
                    images.append(center_image)
                    angles.append(center_angle)
                    #print(np.array(center_image).shape)
                    #images.append(recen)
                    #angles.append(center_angle)
                    images.append(cv2.flip(center_image,1))
                    angles.append(-1*center_angle)
                    #images.append(np.fliplr(recen))
                    #angles.append(-1*center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(X_train.shape)
            #print(y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)
            
            
def process_image(path,index):
    return img


# In[ ]:

print(theano.config.device)
lines=[]
with open('/home/sungmok/udacity/CarND-Behavioral-Cloning-P3-master/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    cnt=0
    for line in reader:
        if cnt>0:
            lines.append(line)
        cnt+=1



train_samples, validation_samples = train_test_split(lines, test_size=0.1)
#validation_samples, test_samples = train_test_split(validation, test_size=0.1)
#print(train_samples.shape)
#print(validation_samples.shape)
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)
#test_generator = generator(test_samples, batch_size=64)
ch, row, col = 3, 64, 64  # Trimmed image format

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,        input_shape=(row, col,ch),output_shape=(row, col,ch)))
#model.add(Cropping2D(cropping=((40,20), (0,0)), input_shape=(row, col,ch)))
model.add(Convolution2D(24,3,3, border_mode='same', activation='relu', subsample=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))       
model.add(Convolution2D(32, 3,3 ,border_mode='valid',subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,3,3,border_mode='valid',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='valid',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object=model.fit_generator(train_generator, samples_per_epoch=             len(train_samples), validation_data=validation_generator,             nb_val_samples=len(validation_samples), nb_epoch=3)
#model.evaluate(test_generator, batch_size=3)
model.save('model.h5')
model.summary()
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
#plt.imsave('cost.png',result)

# In[ ]:




