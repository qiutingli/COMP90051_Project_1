import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


featured_training_df_path = "%s/data/featured_training_df.csv" % os.path.abspath('.')
featured_testing_df_path = "%s/data/featured_testing_df.csv" % os.path.abspath('.')
model_file_path = 'model.sav'
prediction_path = "%s/data/prediction.csv" % os.path.abspath('.')


def load_training_data():
    training_set = pd.read_csv(featured_training_df_path, sep=',')
    X_train = training_set['retweet_or_not', 'num_of_words', 'num_of_hashtags', 'hashtag_contents',
                            'num_of_mentions', 'num_of_urls', 'url_contents']
    y_train = training_set['id']
    return X_train, y_train

def load_testing_data():
    testing_set = pd.read_csv(featured_testing_df_path, sep=',')
    X_test = testing_set
    return X_test

def fit_and_dump_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    # Save the model to disk
    joblib.dump(model, model_file_path)

def make_prediction_file(y_predict):
    pd.DataFrame(y_predict).to_csv(prediction_path)
    prediction_df = pd.read_csv(prediction_path)
    prediction_df.columns = ['Id', 'Predicted']
    prediction_df['Id'] += 1
    prediction_df.to_csv(prediction_path, index=False)


if __name__ == '__main__':
    X_train, y_train = load_training_data()
    X_test = load_testing_data()

    rfc = RandomForestClassifier()
    fit_and_dump_model(X_train, y_train, rfc)

    loaded_model = joblib.load(model_file_path)
    y_predict = loaded_model.predict(X_test)

    make_prediction_file(y_predict)

    # # Compute the training accuracy
    # Accuracy = 0
    # for index in range(len(y_train)):
    # 	current_sample = X_train[index].reshape(1, -1)
    # 	current_label = y_train[index]
    # 	predicted_label = clf.predict(current_sample)
    #
    # 	if current_label == predicted_label:
    # 		Accuracy += 1
    #
    # Accuracy /= len(y_train)
    #
    # # Print stuff
    # print("Classification Accuracy = ", Accuracy)