from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=(5, 5), padding="same", activation='relu', input_shape=(width, height, depth)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=(5, 5), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))

        # softmax classifier
        model.add(Dense(classes, activation='softmax'))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
