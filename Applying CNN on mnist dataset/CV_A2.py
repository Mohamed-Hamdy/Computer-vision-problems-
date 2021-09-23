import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import History 
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import joblib
from keras.utils import np_utils
import tensorflow as tf
nb_classes = 10
input_shape = (28, 28, 1)

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

train_fig = plt.figure(figsize=(14,14))
for i in range(16):
    plt.subplot(4,4,i+1)
    img = plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]),pad=6)
train_fig.savefig("./Real_Data_image.jpg")
#plt.show()
#print("Done")

#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Bulid Model Layers
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(nb_classes, activation="softmax"),
    ]
)


print("compile Model...")
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

print("Fit Model...")
history = model.fit(X_train, Y_train, epochs=15,batch_size=128, validation_data=(X_test, Y_test), validation_split=0.1)

print("Evaluate Model...")
score = model.evaluate(X_test, Y_test, verbose=0)

dict(zip(model.metrics_names, score))
print(history.history.keys())

print("\naccuracy : " , history.history['accuracy'])
print("val_accuracy : " ,history.history['val_accuracy'])
print("\nloss : ",history.history['loss'])
print("val_loss : ",history.history['val_loss'])

# summarize history for accuracy
acc_fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
acc_fig.savefig('./accuracy_figure.png')
        
# summarize history for loss
loss_fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
loss_fig.savefig('./loss_figure.png')

# model summary
print("\nmodel summary")          
print(model.summary())

# Model Architecture
#tf.keras.utils.plot_model(model, to_file='Model_Architecture.png', show_shapes=True)

print('Total Model loss:', score[0])
print('Total Model accuracy:', score[1])
print("\n\n") 
  
# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)
print("number of all predicted Samples :" , len(predicted_classes))

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print("number of correct prediction samples : " , len(correct_indices))
print("number of incorrect prediction samples :" , len(incorrect_indices))

prediction_fig = plt.figure(figsize=(14,14))
for i, correct in enumerate(correct_indices[:16]):
    plt.subplot(4,4,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]),pad=6)
prediction_fig.savefig("./Prediction_image.jpg")

# Save Model 
model.save('model.hdf5')

