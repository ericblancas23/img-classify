
# coding: utf-8

# In[ ]:


import keras 
import numpy as np
from parser import load_data 


# In[ ]:


training_data = load_data('Data/training')
validation_data = load_data('Data/validation')


# In[ ]:


model = Sequential()

model.add(Convolution2D(32,3,3 input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3 input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(62,3,3 input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid')) 



# In[ ]:


model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])


# In[ ]:


model.fit_generator(
training_data,
samples_per_epoch=2048,
nb_epoch,
validation_data = validation_data,
nb_val_samples = 48)

model.save_weights('models/simple_CNN.h5')

