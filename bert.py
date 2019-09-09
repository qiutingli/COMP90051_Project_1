import os
import os.path
import pandas as pd
from collections import defaultdict

from bert_serving.client import BertClient
bc = BertClient(check_length=False)


def bert_train():
    df = pd.DataFrame()
    training_data_path = "./data/train_tweets.txt"
    original_training_data = pd.read_csv(training_data_path, sep="\t", names=['user_id', 'tweet'])
    for tuple in original_training_data.itertuples():
        user_id = tuple[1]
        tweet = tuple[2]
        tweet_vector = bc.encode([tweet])
        user_id_list = [user_id]
        user_id_list.extend(tweet_vector)
        row_df = pd.Series(user_id_list).to_frame().T
        df = pd.concat([df, row_df])
    df.to_csv('./bert_train.csv', index=False)


def bert_test():
    df = pd.DataFrame()
    testing_data_path = "%s/data/test_tweets_unlabeled.txt" % os.path.abspath('.')
    f = open(testing_data_path, "r")
    original_testing_data = pd.DataFrame(f.readlines(), columns=['tweet'])
    f.close()
    for tuple in original_testing_data.itertuples():
        tweet = tuple[1]
        tweet_vector = bc.encode([tweet])
        row_df = pd.Series(tweet_vector).to_frame().T
        df = pd.concat([df, row_df])
    df.to_csv('./bert_test.csv', index=False)


def bert_train2():
    user_tweets = defaultdict(list)
    user_vectors = defaultdict()
    with open("./train_tweets.txt", encoding='UTF-8') as train_tweets:
        for line in train_tweets:
            processed_line = line.strip().split('\t')
            user_id = processed_line[0]
            tweet = processed_line[1]
            user_tweets[user_id].append(tweet)

    for user,tweets in user_tweets.items():
        user_vectors[user] = bc.encode(tweets)

    df = pd.DataFrame()
    for user,vectors in user_vectors.items():
        row_df = pd.DataFrame(vectors)
        row_df = row_df.assign(user_id=[user]*len(row_df))
        df = pd.concat([df, row_df])
    df.to_csv('./bert_train.csv', index=False)


if __name__ == '__main__':
    bert_train()
    bert_test()


bert_df = pd.read_csv('./data/bert.csv')
for tuple in bert_df.itertuples():
    print(tuple[2])
    print(tuple[3].split('\n'))
    break





