import os
import pandas as pd
import stylistic_feature_extractor
from urllib.parse import urlparse
from collections import Counter
from statistics import mode

pd.set_option("display.max_columns", 500)


class FeaturedDataBuilder:
    def __init__(self, original_data, file_path, data_type):
        self.original_data = original_data
        self.file_path = file_path
        self.data_type = data_type
        self.features = ['user_id',
                    'retweet_or_not', 'num_of_words', 'num_of_hashtags', 'most_freq_hashtag',
                    'num_of_mentions', 'num_of_urls', 'most_freq_url', 'num_of_puncts',
                    'num_of_happy_smilies', 'num_of_sad_smilies', 'num_of_emoji', 'num_of_slangs']

    def write_featured_df_row_to_csv(self, featured_df):
        if os.path.exists(self.file_path):
            featured_df.to_csv(self.file_path, mode = 'a', header = False, index = False, encoding = "utf-8")
        else:
            featured_df.to_csv(self.file_path, index = False, encoding = "utf-8")

    def transform_hashtag(self, name, grouped_data):
        result = 'No'
        appended_hashtag_series = grouped_data.get_group(name)['most_freq_hashtag'].apply(lambda x: ''.join(x))
        appended_url_list = list(appended_hashtag_series)
        # for row in appended_list:
        #     print(row.__class__)
        try:
            result = Counter(appended_url_list).most_common(2)[1][0]
        except Exception as e:
            print('hash', e)
        return result

    def transform_url(self, name, grouped_data):
        result = 'No'
        appended_url_series = grouped_data.get_group(name)['most_freq_url'].apply(lambda x: ''.join(x))
        appended_url_list = list(appended_url_series)
        # appended_list = grouped_data.agg({'most_freq_url': 'sum'})['most_freq_url']
        # grouped_data.apply(lambda x: [].extend(x))
        # grouped_data.agg({'b': 'sum', 'c': lambda x: ' '.join(x)})
        try:
            result = Counter(appended_url_list).most_common(2)[1][0]
        except Exception as e:
            print('url', e)
        return result

    def transform_columns(self):
        data = pd.read_csv(self.file_path)
        grouped_data1 = data.groupby('user_id')
        grouped_data2 = data.groupby('user_id')
        # df2 = grouped_data.agg({'most_freq_hashtag': lambda x: ''.join(x), 'most_freq_url': lambda x: ''.join(x)})
        # print(Counter(df2.loc[611]['most_freq_hashtag']).most_common(1)[0])

        for name, group in grouped_data1:
            most_freq_hashtag = self.transform_hashtag(name, grouped_data1)
            data.loc[data.user_id == name, 'most_freq_hashtag'] = most_freq_hashtag

        for name, group in grouped_data2:
            most_freq_url = self.transform_url(name, grouped_data2)
            data.loc[data.user_id == name, 'most_freq_url'] = most_freq_url

        data.to_csv(self.file_path, index = False, encoding = "utf-8")


    def construct_featured_data_frame(self):
        for i in range(len(self.original_data)):
            tweet = self.original_data.iloc[i]['tweet']
            extractor = stylistic_feature_extractor.StylisticFeatureExtractor(tweet)

            retweet_or_not = extractor.determine_retweet()
            num_of_words = extractor.get_num_of_words()
            hashtag_contents = extractor.get_hashtag_contents()
            num_of_hashtags = len(hashtag_contents)
            most_freq_hashtag = hashtag_contents

            num_of_mentions = extractor.get_num_of_mentions()
            most_freq_url = extractor.get_urls()
            num_of_urls = len(most_freq_url)
            num_of_puncts = extractor.get_num_of_puncts()

            num_of_happy_smilies, num_of_sad_smilies = extractor.get_num_of_smilies()
            num_of_emoji = extractor.get_num_of_emoji()
            num_of_slangs = None

            featured_dict = {
                self.features[1]: retweet_or_not,
                self.features[2]: num_of_words,
                self.features[3]: num_of_hashtags,
                self.features[4]: most_freq_hashtag,
                self.features[5]: num_of_mentions,
                self.features[6]: num_of_urls,
                self.features[7]: most_freq_url,
                self.features[8]: num_of_puncts,
                self.features[9]: num_of_happy_smilies,
                self.features[10]: num_of_sad_smilies,
                self.features[11]: num_of_emoji
                # self.featured_df_columns[12]: num_of_slangs
            }
            if self.data_type == 'training':
                featured_dict[self.features[0]] = original_training_data.iloc[i]['user_id']

            featured_df = pd.Series(featured_dict).to_frame().T
            self.write_featured_df_row_to_csv(featured_df)
        self.transform_columns()

        #     featured_df = featured_df.append(featured_dict, ignore_index=True)
        # return featured_df



if __name__ == '__main__':
    # # Read original training dataset
    # training_data_path = "%s/data/train_tweets.txt" % os.path.abspath('.')
    # original_training_data = pd.read_csv(training_data_path, sep ="\t", names = ['user_id', 'tweet'])
    # # Transform original training dataset to featured dataset
    # featured_training_df_path = "%s/data/featured_training_df.csv" % os.path.abspath('.')
    # featured_training_data_builder = FeaturedDataBuilder(original_training_data, featured_training_df_path, "training")
    # featured_training_data_builder.construct_featured_data_frame()
    #
    # # Read original testing dataset
    # testing_data_path = "%s/data/test_tweets_unlabeled.txt" % os.path.abspath('.')
    # f = open(testing_data_path, "r")
    # original_testing_data = pd.DataFrame(f.readlines(), columns = ['tweet'])
    # f.close()
    # # Transform original testing dataset to featured dataset
    # featured_testing_df_path = "%s/data/featured_testing_df.csv" % os.path.abspath('.')
    # featured_testing_data_builder = FeaturedDataBuilder(original_testing_data, featured_testing_df_path, "testing")
    # featured_testing_data_builder.construct_featured_data_frame()

    training_data_path = "%s/data/test_data_for_coding.txt" % os.path.abspath('.')
    original_training_data = pd.read_csv(training_data_path, sep ="\t", names = ['user_id', 'tweet'])
    featured_training_df_path = "%s/data/test_featured_training_df.csv" % os.path.abspath('.')
    featured_training_data_builder = FeaturedDataBuilder(original_training_data, featured_training_df_path, "training")
    featured_training_data_builder.construct_featured_data_frame()





