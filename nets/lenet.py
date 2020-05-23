from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K
import keras.applications.inception_resnet_v2

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initial the model
        model = Sequential()
        inputShape = (height, width, depth)
        keras.preprocessing.image.random_rotation()
