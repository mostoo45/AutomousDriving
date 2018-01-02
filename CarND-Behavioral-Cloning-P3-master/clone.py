import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
tf.python.control_flow_ops = tf

learning_rate=0.001

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
				#images.append(cv2.flip(center_image,1))
				angles.append(-1*center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


lines=[]
with open('/home/sungmok/udacity/CarND-Behavioral-Cloning-P3-master/data/driving_log.csv') as csvfile:
	reader=csv.reader(csvfile)
	cnt=0
	for line in reader:
		if cnt>0:
			lines.append(line)
		cnt+=1
'''
images=[]
measurements=[]
for line in lines:
	source_path=line[0]
	#print(source_path)
	filename=source_path.split('/')[-1]
	current_path='/home/sungmok/udacity/CarND-Behavioral-Cloning-P3-master/data/IMG/'+filename
	image=cv2.imread(current_path)
	images.append(image)
	measurement=float(line[3])
	images.append(cv2.flip(image,1))
	measurements.append(measurement)
	measurements.append(-1.0*measurement)

X_train=np.array(images)
y_train=np.array(measurements)
'''

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print(train_samples)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
model.add(Activation('relu',name='relu2'))
model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1))
model.summary()


'''
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
ch, row, col = 3, 80, 320  # Trimmed image format

model=Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))

#model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5, border_mode='valid', activation='relu', subsample=(2,2)))
model.add(MaxPooling2D(pool_size(2,2),strides=(1,1))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',optimizer=Adam(learning_rate))
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
exit()
'''


