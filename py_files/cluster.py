import numpy as np
from tensorflow.keras import preprocessing as kprocessing
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing


def cluster(corpus, nlp):
    list_of_docs = tokenize(corpus)
    list_of_vec = vectorize(list_of_docs, nlp)

    # Kmeans
    list_of_vec_norm = preprocessing.normalize(list_of_vec)
    kmeans_model = KMeans(n_clusters=6)
    cluster_labels = kmeans_model.fit_predict(list_of_vec_norm)

    series_labels = pd.Series(data=cluster_labels)

    return series_labels


def tokenize(corpus):
    list_corpus = []
    for string in corpus:
        list_words = string.split()
        list_corpus.append(list_words)

    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN",
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(list_corpus)

    return list_corpus


def vectorize(list_of_docs, nlp):
    list_of_vec = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(300)
        vectors = []
        for token in tokens:
            if token in nlp:
                try:
                    vectors.append(nlp[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            list_of_vec.append(avg_vec)
        else:
            list_of_vec.append(zero_vector)

    return list_of_vec


def city_association(df):
    df_ratings = pd.read_csv('data/BX-Book-Ratings.csv')
    df_users = pd.read_csv('data/BX-Users.csv')

    for i in range(6):
        # print("For the cluster number", i, ":")
        df_cluster = df[df['cluster_labels'] == i]
        df_cluster = df_cluster.reset_index()
        df_cluster = df_cluster['isbn']
        df_bad, df_middle, df_good = ratings_association(df_cluster, df_ratings, df_users)
        file_name_bad = "place/bad_ratings_cluster_{cluster}.csv".format(cluster=i)
        file_name_middle = "place/middle_ratings_cluster_{cluster}.csv".format(cluster=i)
        file_name_good = "place/good_ratings_cluster_{cluster}.csv".format(cluster=i)
        df_bad.to_csv(file_name_bad)
        df_middle.to_csv(file_name_middle)
        df_good.to_csv(file_name_good)


def ratings_association(df, df_ratings, df_users):
    df_ratings = pd.merge(df, df_ratings, how='inner', on=['isbn'])
    df_ratings = df_ratings.set_index('isbn')
    df_bad, df_middle, df_good = user_association(df_ratings, df_users)
    return df_bad, df_middle, df_good


def user_association(df_ratings, df_users):
    df_ratings_bad = df_ratings[df_ratings['rating'] <= 3]
    # print("The users that rated the books of this cluster badly (0-3):")
    df_bad = pd.merge(df_users, df_ratings_bad, how='inner', on=['uid'])
    # print(df_bad.to_string())

    df_ratings_middle = df_ratings[(df_ratings['rating'] > 3) & (df_ratings['rating'] <= 7)]
    # print("The users that rated the books of this cluster medium (3-7):")
    df_middle = pd.merge(df_users, df_ratings_middle, how='inner', on=['uid'])
    # print(df_middle.to_string())

    df_ratings_good = df_ratings[df_ratings['rating'] > 7]
    # print("The users that rated the books of this cluster good (7-10):")
    df_good = pd.merge(df_users, df_ratings_good, how='inner', on=['uid'])
    # print(df_good.to_string())

    return df_bad, df_middle, df_good

