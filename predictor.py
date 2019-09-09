import os
import pandas as pd
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier



model_output_file_path = '%s/models/vectorized_200d_sum_knn_5_model.sav' % os.path.abspath('.')
prediction_path = "%s/predictions/vectorized_200d_sum_knn_5_prediction.csv" % os.path.abspath('.')

def load_training_data(training_set_path):
    training_set = pd.read_csv(training_set_path)
    y_train = training_set['user_id']
    X_train = training_set.drop('user_id', axis=1)
    return X_train, y_train

def load_testing_data(testing_set_path):
    testing_set = pd.read_csv(testing_set_path, sep=',')
    # X_test = testing_set.drop('user_id', axis=1)
    X_test = testing_set
    return X_test

def fit_model_and_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    joblib.dump(model, model_output_file_path)
    return y_predict

def make_prediction_file(y_predict):
    pd.DataFrame(y_predict).to_csv(prediction_path)
    prediction_df = pd.read_csv(prediction_path)
    prediction_df.columns = ['Id', 'Predicted']
    prediction_df['Id'] += 1
    prediction_df.to_csv(prediction_path, index=False)

def concat_processed_data():
    train_df1 = pd.read_csv("./data/train_vectorized.csv", sep=',')
    train_df2 = pd.read_csv("./data/train_processed.csv", sep=',')
    train_df1 = train_df1.drop('user_id', axis=1)
    train_joined = pd.concat([train_df2, train_df1], axis=1)
    # train_joined = pd.merge(train_df1, train_df2, on='user_id', how='inner')
    train_joined.to_csv("./data/train_joined.csv", index = False, encoding = "utf-8")

    test_df1 = pd.read_csv("./data/test_vectorized.csv", sep=',')
    test_df2 = pd.read_csv("./data/test_processed.csv", sep=',')
    test_joined = pd.concat([test_df2, test_df1], axis=1)
    test_joined.to_csv("./data/test_joined.csv", index = False, encoding = "utf-8")


if __name__ == '__main__':
    # concat_processed_data()
    training_set_path = "%s/data/train_vectorized_200d_sum.csv" % os.path.abspath('.')
    testing_set_path = "%s/data/test_vectorized_200d_sum.csv" % os.path.abspath('.')
    X_train, y_train = load_training_data(training_set_path)
    X_test = load_testing_data(testing_set_path)

    start_time = datetime.datetime.now()
    # model = RandomForestClassifier()
    model = KNeighborsClassifier(n_neighbors=5)
    y_predict = fit_model_and_predict(X_train, y_train, X_test, model)
    make_prediction_file(y_predict)
    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds
    print('total_time: {} seconds'.format(interval))


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