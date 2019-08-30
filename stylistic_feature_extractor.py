import re
import string
import utility
import tweet_cleaner

# TODO: Extract top 3 most frequent hashtags of each user.  Think what if there's no most frequent hashtag.
#  Extract most frequent web to which url belongs.
class StylisticFeatureExtractor:

    def __init__(self, tweet):
        cleaner = tweet_cleaner.TweetCleaner()
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

    # Return number of punctuations in tweet. TODO: Consider ''
    def get_num_of_puncts(self):
        num = 0
        for char in self.tweet:
            num += 1 if char in string.punctuation else 0
        return num

    # Return number of happy smilies and sad smilies
    def get_num_of_smilies(self):
        num_happy, num_sad = 0, 0
        for char in self.tweet:
            if char in utility.emoticons_happy:
                num_happy += 1
            if char in string.punctuation:
                num_sad += 1
        return num_happy, num_sad

    # Return number of slangs
    def get_num_of_slangs(self):
        return 0

    # Return number of emoji
    def get_num_of_emoji(self):
        return 0


