import glob
import os
import sys
import keras
import cv2
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, Model
import test_cv.utils as utils
import numpy as np
import time
def read_img(img, size):
    """util function to read and preprocess the test image.
    Args:
           img: image.
           size: resize.
    Returns:
           img: original image.
           pimg: processed image.
    """
    pimg = cv2.resize(img, size)
    pimg = np.expand_dims(pimg, axis=0)

    return img, pimg

# 打印输出到log.txt
class Logger(object):
    def __init__(self, filename="./logs/run.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger()

def conv_output(model, layer_name, img):
    """Get the output of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]


batch_size = 128
num_classes = 10
epochs = 12
log_dir = './logs'

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

tensorboard = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(Conv2D(36, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, name='conv_1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          callbacks=[tensorboard],
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# img_path = r'C:\Users\Administrator\Desktop\draw\1.jpg'
img, pimg = read_img(x_test[0].reshape(img_rows, img_cols, 1), size=(28, 28))
pimg = pimg.reshape(1, img_rows, img_cols, 1)
cout = conv_output(model, 'conv_1', pimg)
utils.vis_conv(cout, 6, 'conv_1', 'conv')
# cout = conv_output(model, 'conv_2', pimg)
# utils.vis_conv(cout, 8, 'conv_2', 'conv')

# You can now launch tensorboard with `tensorboard --logdir=./logs` on your
# command line and then go to http://localhost:6006/#projector to view the
# embeddings
