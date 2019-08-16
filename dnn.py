import numpy as np
import tensorflow as tf
from tensorflow import keras
from six.moves import cPickle 
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

train_data_path = 'train.pkl'
validation_data_path = 'val.pkl'
test_data_path = 'test.pkl'

##### Loading data #####
with open(train_data_path, 'rb') as cPickle_file:
	[X_train, y_train] = cPickle.load(cPickle_file)

with open(validation_data_path, 'rb') as cPickle_file:
	[X_val, y_val] = cPickle.load(cPickle_file)

with open(test_data_path, 'rb') as cPickle_file:
	[X_test, y_test] = cPickle.load(cPickle_file)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes=38)
y_val = keras.utils.to_categorical(y_val, num_classes=38)
y_test = keras.utils.to_categorical(y_test, num_classes=38)

model = keras.Sequential([
	keras.layers.Dense(1200, input_dim = 39),
	keras.layers.Activation('relu'),
	keras.layers.Dense(1200),
	keras.layers.Activation('relu'),
	keras.layers.Dense(1200),
	keras.layers.Activation('relu'),
	keras.layers.Dense(1200),
	keras.layers.Activation('relu'),
	keras.layers.Dense(1200),
	keras.layers.Activation('relu'),
	keras.layers.Dense(38),
	keras.layers.Activation('softmax'),
])

rmsprop = keras.optimizers.RMSprop(lr=0.00008, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', cooldown=0, min_lr=0)
callback_lists = [checkpoint, reducelr]
history = model.fit(X_train, y_train, epochs=100, batch_size=64,verbose=1,validation_data=(X_val,y_val),callbacks=callback_lists)

loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

import pickle
file = open('./models/history.pkl', 'wb')
pickle.dump(history.history, file)
file.close()
