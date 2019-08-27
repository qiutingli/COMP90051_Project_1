import os
import pandas as pd
from nltk.tokenize import word_tokenize

pd.set_option("display.max_columns", 500)

class FeatureExtractor:

    def __init__(self, tweet):
        self.tweet = tweet
        self.tokenized_tweet = word_tokenize(tweet)


    # Determine if the tweet is a retweet. Return 1 if it's a retweet, return 0 otherwise.
    def determine_retweet(self):
        if self.tokenized_tweet[0] == 'RT':
            return 1
        else:
            return 0

    # Return number of words in tweet.
    def get_num_of_words(self):

        return len(self.tokenized_tweet)


def initialize_featured_data_frame():
    return pd.DataFrame(columns=['id', 'retweet_or_not', 'num_of_words'])

if __name__ == '__main__':
    training_data_path = "%s/data/sample_data_for_coding.txt" % os.path.abspath('.')
    tweets = pd.read_csv(training_data_path, sep ="\t", names = ['id', 'tweet'])

    featured_df = initialize_featured_data_frame()
    for i in range(len(tweets)):
        tweet = tweets.iloc[i]['tweet']
        extractor = FeatureExtractor(tweet)
        retweet_or_not = extractor.determine_retweet()
        num_of_words = extractor.get_num_of_words()
        featured_dict = {'id': tweets.iloc[i]['id'], 'retweet_or_not': retweet_or_not, 'num_of_words': num_of_words}
        featured_df = featured_df.append(featured_dict, ignore_index = True)

    print(featured_df)


