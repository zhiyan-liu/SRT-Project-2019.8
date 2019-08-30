import numpy as np
import tensorflow as tf
from tensorflow import keras
from six.moves import cPickle 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model

train_data_path = 'train512.pkl'
validation_data_path = 'val512.pkl'
test_data_path = 'test512.pkl'

##### Loading data #####
with open(train_data_path, 'rb') as cPickle_file:
	[X_train, y_train] = cPickle.load(cPickle_file)
	while True:
		try:
			[X, y] = cPickle.load(cPickle_file)
			X_train = np.append(X_train,X,axis=0)
			y_train = np.append(y_train,y)
		except EOFError:
			break

with open(validation_data_path, 'rb') as cPickle_file:
	[X_val, y_val] = cPickle.load(cPickle_file)
	while True:
		try:
			[X, y] = cPickle.load(cPickle_file)
			X_val = np.append(X_val,X,axis=0)
			y_val = np.append(y_val,y)
		except EOFError:
			break

with open(test_data_path, 'rb') as cPickle_file:
	[X_test, y_test] = cPickle.load(cPickle_file)
	while True:
		try:
			[X, y] = cPickle.load(cPickle_file)
			X_test = np.append(X_test,X,axis=0)
			y_test = np.append(y_test,y)
		except EOFError:
			break

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes=41)
y_val = keras.utils.to_categorical(y_val, num_classes=41)
y_test = keras.utils.to_categorical(y_test, num_classes=41)

model = keras.Sequential([
	keras.layers.Dense(1024, input_dim = 576,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('relu'),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1024,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('relu'),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1024,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('relu'),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1024,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('relu'),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1024,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('relu'),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1024,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('relu'),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1024,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('relu'),
	keras.layers.Dense(41,kernel_regularizer=l2(0.003)),
	keras.layers.Activation('softmax'),
])

rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', cooldown=0, min_lr=0)
callback_lists = [checkpoint, reducelr]
history = model.fit(X_train, y_train, epochs=50, batch_size=64,verbose=1,validation_data=(X_val,y_val),callbacks=callback_lists)

loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

import pickle
file = open('./models/history.pkl', 'wb')
pickle.dump(history.history, file)
file.close()
