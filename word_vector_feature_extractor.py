import numpy as np
import os
import zipfile
import pandas as pd
import datetime

from tweet_cleaner import TweetCleaner

pd.set_option("display.max_columns", 500)
vector_dim = 200

def loadGloveModel():
    print("Loading Glove Model")
    f = open("./file_utility/glove.twitter.27B.{}d.txt".format(vector_dim), 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    f.close()
    return model

glove_word_dictionary = loadGloveModel()
# print(model)


class WordVecFeatureExtractor():

    def __init__(self):
        self.model = glove_word_dictionary


    def clean_tokens(self, tweet):
        tweet_cleaner = TweetCleaner()
        tweet_punct_removed = tweet_cleaner.remove_punct(tweet)
        tweet_tokenized = tweet_cleaner.tokenize(tweet_punct_removed)
        tweet_stopwords_removed = tweet_cleaner.remove_stopwords(tweet_tokenized)
        return tweet_stopwords_removed


    def tweet_to_vector(self, tweet):
        cleaned_token_list = self.clean_tokens(tweet)
        print(cleaned_token_list)
        vectors = []
        for token in cleaned_token_list:
            try:
                vectors.append(self.model[token.lower()])
            except Exception as e:
                print('Exception:', e)
        return vectors


    def tweet_to_features_df_mean(self, user_id, tweet):
        vectors = self.tweet_to_vector(tweet)
        length = len(vectors)
        if length != 0:
            features_array = sum(vectors)
        else:
            features_array = np.array([0]*vector_dim)
        features_df = pd.Series(features_array).to_frame().T
        row_df = pd.concat([pd.DataFrame({"user_id": [user_id]}), features_df], axis=1)
        return row_df

    def tweet_to_features_df(self, user_id, tweet):
        vectors = self.tweet_to_vector(tweet)
        length = len(vectors)
        if length != 0:
            features_array = vectors
            # features_array = sum(vectors) / length
        else:
            features_array = np.array([0]*vector_dim)

        df = pd.DataFrame()
        for vector in vectors:
            user_id_list = [str(user_id)]
            user_id_list.extend(vector)
            row_df = pd.Series(user_id_list).to_frame().T
            df = pd.concat([df, row_df])
        df.rename(columns={'0': 'user_id'}, inplace=True)
        return df

    def tweet_to_features_df_for_test_set(self, tweet):
        vectors = self.tweet_to_vector(tweet)
        length = len(vectors)
        if length != 0:
            features_array = sum(vectors)
        else:
            features_array = np.array([0] * vector_dim)
        print(vectors)
        features_df = pd.Series(features_array).to_frame().T
        return features_df



if __name__ == '__main__':

    def write_featured_df_row_to_csv(featured_df, file_path):
        if os.path.exists(file_path):
            featured_df.to_csv(file_path, mode='a', header=False, index=False, encoding="utf-8")
        else:
            featured_df.to_csv(file_path, index=False, encoding="utf-8")

    # training_data_path = "./data/1_sample_training_for_coding.txt"
    # original_training_data = pd.read_csv(training_data_path, sep="\t", names=['user_id', 'tweet'])
    # vectored_training_df_path = "./data/1_sample_training_vectorized_{}d.csv".format(vector_dim)

    training_data_path = "./data/train_tweets.txt"
    original_training_data = pd.read_csv(training_data_path, sep ="\t", names = ['user_id', 'tweet'])
    vectored_training_df_path = "./data/train_vectorized_{}d_sum.csv".format(vector_dim)

    start_time = datetime.datetime.now()

    word_vector_feature_extractor = WordVecFeatureExtractor()
    for row in original_training_data.itertuples(index=False):
        user_id = row[0]
        tweet = row[1]
        row_df = word_vector_feature_extractor.tweet_to_features_df_mean(user_id, tweet)
        write_featured_df_row_to_csv(row_df, vectored_training_df_path)

    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds
    print('total_time: {} seconds'.format(interval))


    # testing_data_path = "%s/data/1_sample_testing_for_coding.txt" % os.path.abspath('.')
    # f = open(testing_data_path, "r")
    # original_testing_data = pd.DataFrame(f.readlines(), columns=['tweet'])
    # f.close()
    # vectored_testing_df_path = "./data/1_sample_testing_vectorized_{}d.csv".format(vector_dim)

    # testing_data_path = "%s/data/test_tweets_unlabeled.txt" % os.path.abspath('.')
    # f = open(testing_data_path, "r")
    # original_testing_data = pd.DataFrame(f.readlines(), columns=['tweet'])
    # f.close()
    # vectored_testing_df_path = "./data/test_vectorized_{}d_sum.csv".format(vector_dim)
    #
    # start_time = datetime.datetime.now()
    #
    # word_vector_feature_extractor = WordVecFeatureExtractor()
    # for row in original_testing_data.itertuples(index=False):
    #     tweet = row[0]
    #     row_df = word_vector_feature_extractor.tweet_to_features_df_for_test_set(tweet)
    #     write_featured_df_row_to_csv(row_df, vectored_testing_df_path)
    #
    # end_time = datetime.datetime.now()
    # interval = (end_time - start_time).seconds
    # print('total_time: {} seconds'.format(interval))

# def sentence2sequence(self, sentence):
    #     glove_wordmap = {}
    #     with open(self.glove_vectors_file, "r") as glove:
    #         for line in glove:
    #             name, vector = tuple(line.split(" ", 1))
    #             glove_wordmap[name] = np.fromstring(vector, sep=" ")
    #
    #     tokens = sentence.lower().split(" ")
    #     rows = []
    #     # words = []
    #
    #     # Greedy search for tokens
    #     for token in tokens:
    #         i = len(token)
    #         while len(token) > 0 and i > 0:
    #             word = token[:i]
    #             if word in glove_wordmap:
    #                 rows.append(glove_wordmap[word])
    #                 # words.append(word)
    #                 token = token[i:]
    #                 i = len(token)
    #             else:
    #                 i = i - 1
    #
    #     features_array = sum(rows)/len(rows)
    #     features_df = pd.Series(features_array).to_frame().T
    #     return features_df
