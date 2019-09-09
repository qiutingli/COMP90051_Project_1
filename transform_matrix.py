import pandas as pd
from bert_serving.client import BertClient

bc = BertClient(check_length=False)

def bert_train():
    df = pd.DataFrame()
    with open("./train_tweets.txt", encoding='UTF-8') as train_tweets:
        for line in train_tweets:
            processed_line = line.strip().split('\t')
            user_id = processed_line[0]
            tweet = processed_line[1]
            tweet_vector = bc.encode([tweet])
            user_id_list = [user_id]
            user_id_list.extend(tweet_vector)
            row_df = pd.Series(user_id_list).to_frame().T
            df = pd.concat([df, row_df])
    df.to_csv('./bert_train.csv', index=False)


def bert_test():
    df = pd.DataFrame()
    with open("./data/1_sample_training_for_coding.txt", encoding='UTF-8') as train_tweets:
        for line in train_tweets:
            processed_line = line.strip().split('\t')
            tweet = processed_line[0]
            tweet_vector = bc.encode([tweet])
            row_df = pd.Series(tweet_vector).to_frame().T
            df = pd.concat([df, row_df])
    df.to_csv('./bert_test.csv', index=False)

bert_test()


original_df = pd.read_csv('./data/bert.csv')
original_df.head()
#
# original_df.to_csv('./data/bert.csv', index = False)
#
# for tuple in original_df.itertuples():
#     user_id = tuple[2]
#     matrix = tuple[3]
#     print(tuple[3])
#     break
#
# for vector in matrix:
#     row = [user_id].append(vector)
#     print(row)

