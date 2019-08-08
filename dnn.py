import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

##### 先导入X_train与y_train #####

y_train = np_utils.to_categorical(y_train, num_classes=38)
model = Sequential([
	Dense(512, input_dim = 26),
	Activation('sigmoid'),
	Dense(425),
	Activation('sigmoid'),
	Dense(512),
	Activation('sigmoid'),
	Dense(38),
	Activation('softmax'),
])

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
