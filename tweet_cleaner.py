import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

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

    def remove_handles(self):
        pass

    def clean_text(self, text):
        # text = self.remove_punct(text)
        text = self.tokenize(text)
        # text = self.remove_stopwords(text)
        # text = self.stemming(text)
        return text

    # df['Tweet_punct'] = df['Tweet'].apply(lambda x: remove_punct(x))
    # df.head(10)

