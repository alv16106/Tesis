from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


def new_neural(dataset):
    data = pd.read_pickle(dataset)

    del data['song_id']
    del data['valence_scaled']
    del data['arousal_scaled']
    del data['std_valence']
    del data['std_arousal']

    y = np.array(data.emotion.tolist())

    del data['emotion']

    # Encode the classification labels
    le = LabelEncoder()
    yy = np_utils.to_categorical(le.fit_transform(y))

    x_train, x_test, y_train, y_test = train_test_split(data, yy, test_size=0.2, random_state = 127)

    num_labels = yy.shape[1]
    filter_size = 2
    num_features = 26
    def build_model_graph():
        model = Sequential()
        model.add(Dense(100, input_dim=num_features, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(160, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_labels, activation='softmax'))
        # Compile the model
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')    
        return model

    model = build_model_graph()
    model.summary()# Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = 100*score[1]
    print('Starting accuracy', accuracy, '%')

    num_epochs = 100
    num_batch_size = 32
    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)


    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: {0:.2%}".format(score[1]))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: {0:.2%}".format(score[1]))

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.save("neuralnet2.h5")


if __name__ == "__main__":
    new_neural('fixed')