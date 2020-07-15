import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import idx2numpy
'''
idx2numpy package provides a tool for converting files to and from IDX format to numpy.ndarray 
'''

X_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
Y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

X_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
Y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')


#print('Shape of the training arrays x-y:', X_train.shape,Y_train.shape,)
#print('Shape of the test arrays x-y :', X_test.shape, Y_test.shape)
print('minimum/maximum values', np.min(X_test),np.max(X_test)) #minimum/maximum values 0 255
#plt.imshow(X_train[0], cmap=plt.cm.binary)
#plt.show()

#normalisation
'''
we Normalise generally to have the same range of values in the inputs
of the artificial neural nets model in order to  guarantee a stable 
convergence of weight and biases.
'''

X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)
#print('minimum/maximum values after normalisation', np.min(X_test),np.max(X_test))# minimum/maximum values after normalisation 0.0 1.0
#plt.imshow(X_train[0], cmap=plt.cm.binary)
#plt.show()

#build the model

model = tf.keras.models.Sequential() #groups a linear stack of layers into a tf.keras.Model
model.add(tf.keras.layers.Flatten()) # to flatten the input
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

#training the model
model.compile(optimizer='',loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train,Y_train, epochs=10)


validation_loss, validation_accuracy = model.evaluate(X_test, Y_test)

print(validation_loss, validation_accuracy)

model.save('handwritten_number_reader.model')
new_model=tf.keras.models.load_model('handwritten_number_reader.model')
predictions= new_model.predict(X_test)
print(predictions)
print(np.argmax(predictions[0]))
plt.imshow(X_test[0])
plt.show()


