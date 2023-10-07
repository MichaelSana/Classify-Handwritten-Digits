# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import tensorflow as tf


def load_dataset():
    # load dataset and seperate into training and testing sets
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    # reshape dataset to have a single channel (grayscale)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

    # one hot encode target values
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    return train_x, train_y, test_x, test_y


def scale_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # return normalized images
    return train_norm, test_norm


def define_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
              kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
              kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
              kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu',
              kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    kfold = sklearn.model_selection.KFold(n_folds,
                                          shuffle=True,
                                          random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):

        model = define_model()
        # select rows for train and test
        train_x, train_y, test_x, test_y = dataX[train_ix], dataY[train_ix], \
            dataX[test_ix], dataY[test_ix]

        history = model.fit(train_x, train_y, epochs=10, batch_size=32,
                            validation_data=(test_x, test_y), verbose=0)
        # evaluate model
        _, acc = model.evaluate(test_x, test_y, verbose=0)
        print('> %.3f' % (acc * 100.0))

        scores.append(acc)
        histories.append(history)
    return scores, histories


def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'],
                 color='blue',
                 label='train')
        plt.plot(histories[i].history['val_loss'],
                 color='orange',
                 label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'],
                 color='blue',
                 label='train')
        plt.plot(histories[i].history['val_accuracy'],
                 color='orange',
                 label='test')
    plt.show()


def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


def run_test_harness():
    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()
    # prepare pixel data
    train_x, test_x = scale_pixels(train_x, test_x)
    # evaluate model
    scores, histories = evaluate_model(train_x, train_y)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


# entry point, run the test harness
run_test_harness()
