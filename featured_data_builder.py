import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

pd.set_option("display.max_columns", 500)


class TweetCleaner:

    def __init__(self):
        pass

    # Remove punctuation in tweet.
    def remove_punct(self, text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text

    def tokenize(self, text):
        text = word_tokenize(text)
        return text

    def remove_stopwords(self, text):
        stopword = stopwords.words('english')
        text = [word for word in text if word not in stopword]
        return text

    def stemming(self, text):
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text]
        return text

    def clean_text(self, text):
        text = self.remove_punct(text)
        text = self.tokenize(text)
        text = self.remove_stopwords(text)
        text = self.stemming(text)
        return text

    # df['Tweet_punct'] = df['Tweet'].apply(lambda x: remove_punct(x))
    # df.head(10)


class FeatureExtractor:

    def __init__(self, tweet):
        cleaner = TweetCleaner()
        self.tweet = tweet
        self.cleaned_tweet = cleaner.clean_text(tweet)


    # Determine if the tweet is a retweet. Return 1 if it's a retweet, return 0 otherwise.
    def determine_retweet(self):
        return 1 if self.cleaned_tweet[0] == 'RT' else 0

    # Return number of words in tweet.
    def get_num_of_words(self):
        return len(self.cleaned_tweet)

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
    # TODO: Generalize it to fit both training and testing data
    def __init__(self, original_data, file_path, type):

        self.original_data = original_data
        self.file_path = file_path
        self.type = type
        features = ['retweet_or_not', 'num_of_words', 'num_of_hashtags', 'hashtag_contents',
                                        'num_of_mentions', 'num_of_urls', 'url_contents']
        if self.type == 'training':
            self.featured_df_columns = features + ['id']
        elif self.type == "testing":
            self.featured_df_columns = features


    # Initialize featured data frame by using given column names.
    def initialize_featured_data_frame(self):
        return pd.DataFrame(columns = self.featured_df_columns)


    # Construct featured data frame iterating the original data.
    def construct_featured_data_frame(self):
        for i in range(len(self.original_data)):
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
                self.featured_df_columns[0]: retweet_or_not,
                self.featured_df_columns[1]: num_of_words,
                self.featured_df_columns[2]: num_of_hashtags,
                self.featured_df_columns[3]: 0, #hashtag_contents, TODO: Find ways dealing with it
                self.featured_df_columns[4]: num_of_mentions,
                self.featured_df_columns[5]: num_of_urls,
                self.featured_df_columns[6]: 0 #url_contents
            }
            if self.type == 'training':
                featured_dict[self.featured_df_columns[7]] = original_training_data.iloc[i]['id']

            featured_df = pd.Series(featured_dict).to_frame().T
            self.write_featured_df_to_csv(featured_df)

        #     featured_df = featured_df.append(featured_dict, ignore_index=True)
        # return featured_df


    def write_featured_df_to_csv(self, featured_df):
        if os.path.exists(self.file_path):
            featured_df.to_csv(self.file_path, mode = 'a', header = False, index = False, encoding = "utf-8")
        else:
            featured_df.to_csv(self.file_path, index = False, encoding = "utf-8")



if __name__ == '__main__':

    # Read original datasets
    training_data_path = "%s/data/train_tweets.txt" % os.path.abspath('.')
    original_training_data = pd.read_csv(training_data_path, sep ="\t", names = ['id', 'tweet'])

    testing_data_path = "%s/data/test_tweets_unlabeled.txt" % os.path.abspath('.')
    # original_testing_data = pd.read_csv(testing_data_path, names = ['tweet'])
    f = open(testing_data_path, "r")
    original_testing_data = pd.DataFrame(f.readlines(), columns = ['tweet'])
    f.close()

    # # Transform original training dataset to featured dataset
    # featured_training_df_path = "%s/data/featured_training_df.csv" % os.path.abspath('.')
    # featured_training_data_builder = FeaturedDataBuilder(original_training_data, featured_training_df_path, "training")
    # featured_df = featured_training_data_builder.construct_featured_data_frame()

    # Transform original testing dataset to featured dataset
    featured_testing_df_path = "%s/data/featured_testing_df.csv" % os.path.abspath('.')
    featured_testing_data_builder = FeaturedDataBuilder(original_testing_data, featured_testing_df_path, "testing")
    featured_testing_data_builder.construct_featured_data_frame()
