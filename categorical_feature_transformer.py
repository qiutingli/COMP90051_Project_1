import pandas as pd
import numpy as np
import os
import gc, copy
from gensim.models import Word2Vec # categorical feature to vectors
from random import shuffle
import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", 500)


categorical_cols = ['most_freq_hashtag', 'most_freq_url']

print('Loading data')
featured_training_df_path = "%s/data/train_featured.csv" % os.path.abspath('.')
featured_testing_df_path = "%s/data/test_featured.csv" % os.path.abspath('.')
train_df = pd.read_csv(featured_training_df_path, sep =",")
test_df = pd.read_csv(featured_testing_df_path, sep =",")

missing_cols = set(train_df.columns) - set(test_df.columns)
for c in missing_cols:
    test_df[c] = 0

# Ensure the order of column in the test set is in the same order than in train set
test_df = test_df[train_df.columns]
full_df = pd.concat([train_df, test_df])

# print('Daset shape rows:{} cols:{}'.format(*full_df.shape))
# del train_df
# del test_df
# gc.collect()

def apply_w2v(sentences, model, num_features):
    def _average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0.
        for word in words:
            if word in vocabulary:
                n_words = n_words + 1.
                feature_vector = np.add(feature_vector, model[word])

        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector

    vocab = set(model.wv.index2word)
    feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
    return np.array(feats)


def gen_cat2vec_sentences(data):
    X_w2v = copy.deepcopy(data)
    names = list(X_w2v.columns.values)
    for c in names:
        X_w2v[c] = X_w2v[c].fillna('unknow').astype('category')
        X_w2v[c].cat.categories = ["%s %s" % (c,g) for g in X_w2v[c].cat.categories]
        print(X_w2v[c].cat.categories)
    X_w2v = X_w2v.values.tolist()
    return X_w2v


print('Cat2Vec...')
n_cat2vec_feature  = len(categorical_cols) # define the cat2vecs dimentions
n_cat2vec_window   = len(categorical_cols) * 2 # define the w2v window size


def fit_cat2vec_model():
    X_w2v = gen_cat2vec_sentences(full_df.loc[:, categorical_cols].sample(frac=0.6))
    for i in X_w2v:
        shuffle(i)
    model = Word2Vec(X_w2v, size=n_cat2vec_feature, window=n_cat2vec_window)
    return model

print('Fit cat2vec model')
c2v_model = fit_cat2vec_model()


print('apply_w2v for cat2vec')
train_objs_num = len(train_df)
tr_c2v_matrix = apply_w2v(gen_cat2vec_sentences(full_df[:train_objs_num][categorical_cols]), c2v_model, n_cat2vec_feature)
te_c2v_matrix = apply_w2v(gen_cat2vec_sentences(full_df[train_objs_num:][categorical_cols]), c2v_model, n_cat2vec_feature)

len(tr_c2v_matrix)
len(te_c2v_matrix)

train_preprocessed = full_df[:train_objs_num]
test_preprocessed = full_df[train_objs_num:]

train_preprocessed[categorical_cols] = tr_c2v_matrix
test_preprocessed[categorical_cols] = te_c2v_matrix

train_preprocessed_path = "%s/data/train_processed.csv" % os.path.abspath('.')
train_preprocessed.to_csv(train_preprocessed_path, index = False, encoding = "utf-8")
test_preprocessed_path = "%s/data/test_processed.csv" % os.path.abspath('.')
test_preprocessed.to_csv(test_preprocessed_path, index = False, encoding = "utf-8")


# featured_training_df_path = "%s/data/train_featured.csv" % os.path.abspath('.')
# featured_testing_df_path = "%s/data/test_featured.csv" % os.path.abspath('.')
#
# train_df = pd.read_csv(featured_training_df_path)
# test_df = pd.read_csv(featured_testing_df_path)
#
# missing_cols = set( train_df.columns ) - set( test_df.columns )
# for c in missing_cols:
#     test_df[c] = 0
#
# # Ensure the order of column in the test set is in the same order than in train set
# test_df = test_df[train_df.columns]
# test_df.head()
# full_df = pd.concat([train_df, test_df])
# full_df.head()
#
# train_objs_num = len(train_df)
# dataset_preprocessed = pd.get_dummies(full_df)
# train_preprocessed = dataset_preprocessed[:train_objs_num]
# test_preprocessed = dataset_preprocessed[train_objs_num:]


# df = pd.read_csv(test_preprocessed_path)
# for col_name, col in df_trn.items():
#     if is_string_dtype(col):
#         df_trn[col_name] = col.astype('category').cat.as_ordered()
#
# for n,c in dftest.items():
#     if (n in dftrn.columns) and (dftrn[n].dtype.name=='category'):
#         dftest[n] = pd.Categorical(c, categories=df_trn[n].cat.categories, ordered=True)






