import math
from tabnanny import verbose
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN
import data_load as dl
import dataset_visual as dv

def mnist_fnn(x_train, y_train, x_test, y_test):
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]
    n_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    inputs = tf.keras.Input(shape=(28, 28, 1), name='data')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='lenet')

    #model.summary()

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )
        
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs\\", histogram_freq=1)

    model.fit(x_train, y_train, batch_size=100, epochs=30, verbose=0)
    eval_value = model.evaluate(x_test, y_test)
    return eval_value

"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
value = mnist_fnn(x_train, y_train, x_test, y_test)
print(value)
"""

"""
imgpath = "../datas_modified/mnist/train/"+ str(0)
imgfiles = os.listdir(imgpath)


for file in imgfiles:
    img = imageio.imread(imgpath + "/" + file)
    img = (img / 255)
    img = img.reshape((1,28,28))

    predict = model.predict(img)
    predict=np.argmax(predict,axis=1)   
    print(predict)
"""

def mnist_rnn(x_train, y_train, x_test, y_test):
    n_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    model = Sequential()
    model.add(SimpleRNN(input_dim=28, input_length=28, units=50, unroll=True))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))

    #model.summary()

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )
        

    model.fit(x_train, y_train, batch_size=100, epochs=30, verbose=0)
    eval_value = model.evaluate(x_test, y_test)
    
    return eval_value



def mnist_cnn(x_train, y_train, x_test, y_test):
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    n_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    inputs = tf.keras.Input(shape=(28, 28, 1), name='data')
    x = tf.keras.layers.Conv2D(2, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
    x = tf.keras.layers.Conv2D(2, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(120, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='lenet')

    #model.summary()

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )
        
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs\\", histogram_freq=1)

    model.fit(x_train, y_train, batch_size=100, epochs=30, verbose=0)
    eval_value = model.evaluate(x_test, y_test)
    
    return eval_value

"""
dv.make_mnist(True)

(x_train, y_train), (x_test, y_test) = dl.load_mnist(flag = True, path='../datas/mnist/train/')
print(np.shape(x_train))
eval_value_o = mnist_rnn(x_train, y_train, x_test, y_test)
print(eval_value_o)
"""


def p_mnist_nn(x_train, y_train, x_test, y_test):
    loss_f = []
    acc_f = []
    loss_c = []
    acc_c = []
    loss_r = []
    acc_r = []
    for iter in range(1):
        eval_value_n = mnist_fnn(x_train, y_train, x_test, y_test)
        loss_f.append(eval_value_n[0])
        acc_f.append(eval_value_n[1])

        eval_value_n = mnist_cnn(x_train, y_train, x_test, y_test)
        loss_c.append(eval_value_n[0])
        acc_c.append(eval_value_n[1])

        eval_value_n = mnist_rnn(x_train, y_train, x_test, y_test)
        loss_r.append(eval_value_n[0])
        acc_r.append(eval_value_n[1])
    return loss_f, acc_f, loss_r, acc_r, loss_c, acc_c