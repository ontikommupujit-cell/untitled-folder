import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)
datagen=ImageDataGenerator(rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)
datagen.fit(x_train)
model=models.Sequential([
layers.Flatten(input_shape=(28,28)),
layers.Dense(128,activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)),
layers.Dense(64,activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)),
layers.Dense(10,activation='softmax')
])
model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(datagen.flow(x_train,y_train,batch_size=32),epochs=10)
test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"Test accuracy: {test_acc}")
predictions=model.predict(x_test)
plt.imshow(x_test[0].reshape(28,28),cmap=plt.cm.binary)
plt.title(f"Predicted: {predictions[0].argmax()}")
plt.show()
