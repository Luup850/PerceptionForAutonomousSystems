import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


class CNN:

    def __init__(self):
        self.classes = ["Book", "Box", "Cup", "Nothing"]
        self.paths = ["D:\\DataTraining\\book\\*", "D:\\DataTraining\\box\\*","D:\\DataTraining\\cup\\*", "D:\\DataTraining\\nothing\\*"]
        self.X = []
        self.y = []
        self.model = object


    def _prep_data(self):
    # Prepare data
        for i,p in enumerate(self.paths):
            for filename in glob.glob(p):
                img = cv2.imread(filename)
                img = cv2.resize(img, (32, 32))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                self.X.append(img)
                self.y.append(i)

        #print("Images: ", len(X))

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def train(self):
        self._prep_data()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)


        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
        self.model.add(layers.MaxPooling2D((7, 7)))
        #model.add(layers.Conv2D(4, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(4, activation='softmax'))

        self.model.summary()
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        #model.load_weights("D:\DataTraining\model.h5")
        #xtest = tf.expand_dims(X_test[3], axis=-1)
        #print(np.shape(X_test[3:4]))
        #stuff = model.predict(X_test[3:4])
        #stuff = model(X_test[3], training=False)
        #s = y_test[3]
        #print("s")

    def save(self):
        self.model.save_weights("D:\DataTraining\model.h5")

    def load(self, path):
        self.model.load_weights(path)

    def predict(self, img):
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[np.newaxis, :, :,]
        #stuff = self.model.predict(img)
        index = np.argmax(self._model.predict(img, batch_size=len(img)), axis = 1)

        return self.classes[index[0]]

    
cnn = CNN()

cnn.train()
    #model.save('path/to/location')
    #model = keras.models.load_model('path/to/location')

# %%
