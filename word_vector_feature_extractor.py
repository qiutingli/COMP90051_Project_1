import numpy as np
import os
import zipfile
import pandas as pd
from six.moves.urllib.request import urlretrieve

class WordVecFeatureExtractor():

    def __init__(self):
        self.glove_zip_file = "%s/file_utility/glove.6B.zip" % os.path.abspath('.')
        self.glove_vectors_file = "%s/file_utility/glove.twitter.27B.50d.txt" % os.path.abspath('.')

    def unzip_single_file(self, zip_file_name, output_file_name):
        """
            If the outFile is already created, don't recreate
            If the outFile does not exist, create it from the zipFile
        """
        if not os.path.isfile(output_file_name):
            with open(output_file_name, 'wb') as out_file:
                with zipfile.ZipFile(zip_file_name) as zipped:
                    for info in zipped.infolist():
                        if output_file_name in info.filename:
                            with zipped.open(info) as requested_file:
                                out_file.write(requested_file.read())
                                return


    def sentence2sequence(self, sentence):
        # if (not os.path.isfile(self.glove_zip_file) and
        #         not os.path.isfile(self.glove_vectors_file)):
        #     urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip",
        #                 self.glove_zip_file)
        self.unzip_single_file(self.glove_zip_file, self.glove_vectors_file)

        glove_wordmap = {}
        with open(self.glove_vectors_file, "r") as glove:
            for line in glove:
                name, vector = tuple(line.split(" ", 1))
                glove_wordmap[name] = np.fromstring(vector, sep=" ")

        tokens = sentence.lower().split(" ")
        rows = []
        words = []
        # Greedy search for tokens
        for token in tokens:
            i = len(token)
            while len(token) > 0 and i > 0:
                word = token[:i]
                if word in glove_wordmap:
                    rows.append(glove_wordmap[word])
                    words.append(word)
                    token = token[i:]
                    i = len(token)
                else:
                    i = i - 1
        features_array = sum(rows)/len(rows)
        features_df = pd.Series(features_array).to_frame().T
        return features_df

# if __name__ == '__main__':
#     sentence = "The quick brown fox jumped over the lazy dog."
#     array = WordVecFeatureExtractor().sentence2sequence(sentence)
#     df = pd.Series(array).to_frame().T
#     pd.concat([df, df], axis=1)
