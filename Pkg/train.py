import tensorflow as tf
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

class DnCnn_Class_Train:
    def __init__(self):
        print("Constructor Called")

        # Load dataset from HDF5
        with h5py.File("../Datapy/dataset.h5", "r") as h5f:
            self.X_TRAIN = np.array(h5f["X_train"])
            self.Y_TRAIN = np.array(h5f["Y_train"])
            self.X_EVALUATE = np.array(h5f["X_val"])
            self.Y_EVALUATE = np.array(h5f["Y_val"])

        print(f"Train Data: {self.X_TRAIN.shape}, Validation Data: {self.X_EVALUATE.shape}")

    def ModelMaker(self):
        self.myModel = Sequential()

        # First Conv layer
        self.myModel.add(Conv2D(64, (3,3), padding='same', input_shape=(60,60,3),
                                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)))
        self.myModel.add(Activation('relu'))

        # Middle Layers
        for _ in range(18):
            self.myModel.add(Conv2D(64, (3,3), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)))
            self.myModel.add(BatchNormalization())
            self.myModel.add(Activation('relu'))

        # Last Conv Layer
        self.myModel.add(Conv2D(3, (3,3), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)))

        # Compile model
        self.myModel.compile(optimizer=Adam(learning_rate=0.0002), loss="mean_squared_error")
        print("Model Created")
        self.myModel.summary()

    def trainModel(self, batch_size, epochs):
        self.myModel.fit(self.X_TRAIN, self.Y_TRAIN, batch_size=batch_size, epochs=epochs,
                         validation_data=(self.X_EVALUATE, self.Y_EVALUATE))

        # Save model
        self.myModel.save("DnCNN_Model.h5")
        print("Model trained and saved!")

# Train the model
DnCNN = DnCnn_Class_Train()
DnCNN.ModelMaker()
DnCNN.trainModel(batch_size=100, epochs=50)
