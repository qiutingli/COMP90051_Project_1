import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model

from sklearn.model_selection import train_test_split


def load_training_data(training_set_path):
    training_set = pd.read_csv(training_set_path)
    # train, test = train_test_split(training_set, test_size=0.2)
    y_train = training_set['user_id']
    X_train = training_set.drop('user_id', axis=1)
    # y_validate = test['user_id'].to_frame()
    # X_validate = test.drop('user_id', axis=1)
    return X_train, y_train

def load_testing_data(testing_set_path):
    testing_set = pd.read_csv(testing_set_path, sep=',')
    # X_test = testing_set.drop('user_id', axis=1)
    X_test = testing_set
    return X_test

def make_prediction_file(y_predict):
    prediction_path = "./predictions/prediction_keras_sequential_200d.csv"
    prediction_df = pd.DataFrame(y_predict, columns=['Predicted'])
    prediction_df.index += 1
    prediction_df.to_csv(prediction_path, index_label='Id')

def get_labels():
    labeled_users = []
    labels = []
    labels_index = {}
    label = 0
    # for index, row in y_train.iterrows():
    for row in y_train:
        # user = row['user_id']
        user = row
        if user not in labeled_users:
            labels_index[label] = user
            labeled_users.append(user)
            label = label + 1
        for (key, value) in labels_index.items():
            if value == user:
                labels.append(key)
    return labels, labels_index

def transform_prediction_back_to_user_id(y_predict, labels_index):
    predictions = [labels_index[x] for x in y_predict]
    print(labels_index)
    return predictions

if __name__ == '__main__':
    training_set_path = "./data/train_vectorized_200d.csv"
    testing_set_path = "./data/test_vectorized_200d.csv"
    X_train, y_train = load_training_data(training_set_path)
    X_test = load_testing_data(testing_set_path)

    labels, labels_index = get_labels()
    num_classes = len(labels_index)
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=num_classes)
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=200))
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Dense(len(one_hot_labels[0])))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, one_hot_labels, epochs=20, batch_size=256)
    model.save("./models/keras_sequential_200d.h5")
    # score = model.evaluate(X_train, one_hot_labels, batch_size=32)
    # print(score)
    y_predict = model.predict_classes(X_test)
    print('prediction: ', y_predict)
    predictions = transform_prediction_back_to_user_id(y_predict, labels_index)
    print('transformed_prediction: ', predictions)
    make_prediction_file(predictions)


    model = load_model("./models/keras_sequential_200d.h5")
    y_validate = model.predict_classes(X_test)
    make_prediction_file(y_validate)

