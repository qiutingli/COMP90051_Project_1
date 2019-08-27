import os
import pandas as pd
import re
from nltk.tokenize import word_tokenize

pd.set_option("display.max_columns", 500)


class FeatureExtractor:

    def __init__(self, tweet):
        self.tweet = tweet
        self.tokenized_tweet = word_tokenize(tweet)


    # Determine if the tweet is a retweet. Return 1 if it's a retweet, return 0 otherwise.
    def determine_retweet(self):
        return 1 if self.tokenized_tweet[0] == 'RT' else 0

    # Return number of words in tweet.
    def get_num_of_words(self):
        return len(self.tokenized_tweet)

    # Return the  hash tag contents in tweet.
    def get_hashtag_contents(self):
        return re.findall(r"#(\w+)", self.tweet)

    # Return number of mentions in tweet. TODO: Double check.
    def get_num_of_mentions(self):
        return len(re.findall(r"@(\w+)", self.tweet))

    # Determine if the tweet contains urls.
    def get_urls(self):
        urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', self.tweet)
        return urls



class FeaturedDataBuilder:

    def __init__(self, original_data):
        self.featured_df_columns = ['id',
                                    'retweet_or_not', 'num_of_words', 'num_of_hashtags', 'hashtag_contents',
                                    'num_of_mentions', 'num_of_urls', 'url_contents']
        self.original_data = original_data


    # Initialize featured data frame by using given column names.
    def initialize_featured_data_frame(self):
        return pd.DataFrame(columns = self.featured_df_columns)


    # Construct featured data frame iterating the original data.
    def construct_featured_data_frame(self):
        featured_df = self.initialize_featured_data_frame()
        for i in range(len(self.original_data))[1:10]:
            tweet = self.original_data.iloc[i]['tweet']
            extractor = FeatureExtractor(tweet)

            retweet_or_not = extractor.determine_retweet()
            num_of_words = extractor.get_num_of_words()
            hashtag_contents = extractor.get_hashtag_contents()
            num_of_hashtags = len(hashtag_contents)
            num_of_mentions = extractor.get_num_of_mentions()
            url_contents = extractor.get_urls()
            num_of_urls = len(url_contents)

            featured_dict = {
                self.featured_df_columns[0]: original_training_data.iloc[i]['id'],
                self.featured_df_columns[1]: retweet_or_not,
                self.featured_df_columns[2]: num_of_words,
                self.featured_df_columns[3]: num_of_hashtags,
                self.featured_df_columns[4]: hashtag_contents,
                self.featured_df_columns[5]: num_of_mentions,
                self.featured_df_columns[6]: num_of_urls,
                self.featured_df_columns[7]: url_contents
            }
            featured_df = featured_df.append(featured_dict, ignore_index=True)

        self.featured_df = featured_df
        return self.featured_df


    def write_featured_df_to_csv(self, file):
        if os.path.exists(file):
            self.featured_df.to_csv(file)
        else:
            try:
                os.makedirs(file)
            except Exception as e:
                print(e)
            self.featured_df.to_csv(file, index=False, encoding="utf-8")



if __name__ == '__main__':

    # Read original training dataset
    training_data_path = "%s/data/train_tweets.txt" % os.path.abspath('.')
    original_training_data = pd.read_csv(training_data_path, sep ="\t", names = ['id', 'tweet'])

    # Transform original training dataset to featured dataset
    featured_data_builder = FeaturedDataBuilder(original_training_data)
    featured_df = featured_data_builder.construct_featured_data_frame()
    featured_training_df_path = "%s/data/featured_training_df.csv" % os.path.abspath('.')
    featured_data_builder.write_featured_df_to_csv(featured_training_df_path)
