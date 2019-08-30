import os
import pandas as pd
import stylistic_feature_extractor

pd.set_option("display.max_columns", 500)


class FeaturedDataBuilder:
    # TODO: Generalize it to fit both training and testing data
    def __init__(self, original_data, file_path, type):

        self.original_data = original_data
        self.file_path = file_path
        self.type = type
        self.features = ['user_id',
                    'retweet_or_not', 'num_of_words', 'num_of_hashtags', 'most_freq_hashtag',
                    'num_of_mentions', 'num_of_urls', 'most_freq_url', 'num_of_puncts',
                    'num_of_happy_smilies', 'num_of_sad_smilies', 'num_of_emoji', 'num_of_slangs']

    def write_featured_df_row_to_csv(self, featured_df):
        if os.path.exists(self.file_path):
            featured_df.to_csv(self.file_path, mode = 'a', header = False, index = False, encoding = "utf-8")
        else:
            featured_df.to_csv(self.file_path, index = False, encoding = "utf-8")

    def transform_columns(self):
        pass

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
            url_contents = extractor.get_urls()
            num_of_urls = len(url_contents)
            most_freq_url = None
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
                # self.features[7]: most_freq_url,
                self.features[8]: num_of_puncts,
                self.features[9]: num_of_happy_smilies,
                self.features[10]: num_of_sad_smilies,
                self.features[11]: num_of_emoji
                # self.featured_df_columns[12]: num_of_slangs
            }
            if self.type == 'training':
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

data = pd.read_csv(featured_training_df_path)
data.head()
grouped_data = data.groupby('user_id')

for name, group in grouped_data:
    print(name)
    print(grouped_data.get_group(name)['num_of_hashtags'].apply(lambda x: list(str(x))).sum())









