from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import sys, os
from keras.callbacks import TensorBoard
import keras_applications
log_dir = './logs'
# 最多使用的单词数
max_features = 20000
# 循环神经网络的截断长度
maxlen = 80
batch_size = 32

# 打印输出到log.txt
class Logger(object):
    def __init__(self, filename="./logs/run_2epoch.log"):
        self.terminal = sys.stdout
        if os.path.exists('./logs/'):
            self.log = open(filename, "a", encoding="utf-8")
        else:
            os.makedirs('./logs/')
            self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger()

(trainX, trainY), (testX, testY) = imdb.load_data(num_words=max_features)
print(len(trainX), 'train_sequences')
print(len(testX), 'test_sequences')


trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
testX = sequence.pad_sequences(testX, maxlen=maxlen)

print('trainX shape:', trainX.shape)
print('testX shape:', testX.shape)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=batch_size, epochs=2, validation_data=(testX, testY), callbacks=[TensorBoard(log_dir=log_dir)])

score = model.evaluate(testX, testY, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
