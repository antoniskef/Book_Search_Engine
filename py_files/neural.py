import cluster
import pandas as pd
import numpy as np
import re
import nltk
import gensim.downloader as gensim_api
from tensorflow.keras import preprocessing as kprocessing
from tensorflow.keras import models
from tensorflow.keras import layers


class Neural:

    def neural(self):
        df_books = pd.read_csv('data/Final_BX-Books.csv')
        df_books_train = pd.read_csv('data/Final_BX-Books.csv')
        df_books_test = pd.read_csv('data/Final_BX-Books.csv')

        ratings = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        df_books_train = df_books_train[df_books_train["user_rating"].isin(ratings)]
        df_books_test = df_books_test[~ df_books_test["user_rating"].isin(ratings)]

        # nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words("english")

        df_books["clean_summary"] = df_books["summary"].apply(lambda n: self.preprocess_text(n, stopwords=stopwords))
        df_books_train["clean_summary"] = df_books_train["summary"].apply(lambda n: self.preprocess_text(n, stopwords=stopwords))
        df_books_test["clean_summary"] = df_books_test["summary"].apply(lambda n: self.preprocess_text(n, stopwords=stopwords))

        # pre trained model
        nlp = gensim_api.load("word2vec-google-news-300")

        corpus = df_books["clean_summary"]
        corpus_train = df_books_train["clean_summary"]
        corpus_test = df_books_test["clean_summary"]

        vocabulary = self.create_vocabulary(corpus)

        X_train = self.create_x(corpus_train)
        X_test = self.create_x(corpus_test)

        # print(vocabulary)

        Y_train = np.array(df_books_train["user_rating"].values)
        Y_train = Y_train.astype(int)
        t = np.array(Y_train).reshape(-1)
        t = np.eye(11)[t]
        Y_train = t

        # dhmioyrgoyme ena numpy array me arithmo grammwn oso to plhthos twn leksewn sto leksiko kai
        # arithmo sthlwn iso me 300, afoy to kathe dianysma poy anaparista mia leksh exei mhkos iso
        # me 300 symfvna me to pre trained montelo poy fortwsame parapanw.

        embeddings = np.zeros((len(vocabulary) + 1, 300))
        for word, idx in vocabulary.items():
            try:
                embeddings[idx] = nlp[word]
            # an h leksh den yparxei sto montelo apla menoyn ta mhdenika poy mphkan kata thn arxikopoihsh.
            except:
                print("no such word")

        ###### neural network
        x_in = layers.Input(shape=(30,))
        x = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings], input_length=30, trainable=False)(x_in)
        x = layers.Dense(20, activation='relu')(x)
        x = layers.Flatten()(x)
        x_out = layers.Dense(11, activation='softmax')(x)

        model = models.Model(x_in, x_out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        # train
        model.fit(x=X_train, y=Y_train, validation_split=0.3, batch_size=8, epochs=10, shuffle=True, verbose=0)

        # predict
        predictions = model.predict(x=X_test)
        rows, cols = predictions.shape
        pred = np.array([])
        for i in range(rows):
            m = np.argmax(predictions[i])
            m = int(m)
            pred = np.append(pred, m)
        # print(pred)

        index_test = df_books_test['isbn']
        df_new_ratings = pd.DataFrame(data=pred, index=index_test, columns=['user_rating'])

        df_old_ratings = df_books_train.loc[:, ["isbn", "user_rating"]]
        df_old_ratings = df_old_ratings.set_index('isbn')
        # print(df_old_ratings)

        df_ratings = pd.concat([df_old_ratings, df_new_ratings], sort=False)
        # print(df_ratings)

        df_books = pd.read_csv('data/Final_BX-Books.csv')
        df_books = df_books.drop("user_rating", axis=1)
        # print(df_books)
        df_books = df_books.set_index('isbn')
        df_final = pd.merge(df_books, df_ratings, how='left', on=['isbn'])
        # print(df_final)

        cluster_labels = cluster.cluster(corpus, nlp)
        df_final = df_final.assign(cluster_labels=cluster_labels.values)

        df_final.to_csv('data/Final_BX-Books.csv')

        cluster.city_association(df_final)

    @staticmethod
    def create_vocabulary(corpus):
        # dhmioyrgoyme lista poy exei listes me tis lekseis toy kathe keimenoy
        list_corpus = []
        for string in corpus:
            list_words = string.split()
            list_corpus.append(list_words)

        # tokenize text - dhmioyrgoyme leksiko me oles tis lekseis poy yparxoyn sta keimena
        tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN",
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(list_corpus)
        vocabulary = tokenizer.word_index

        return vocabulary

    @staticmethod
    def create_x(corpus):
        # dhmioyrgoyme lista poy exei listes me tis lekseis toy kathe keimenoy
        list_corpus = []
        for string in corpus:
            list_words = string.split()
            list_corpus.append(list_words)

        # tokenize text - dhmioyrgoyme leksiko me oles tis lekseis poy yparxoyn sta keimena
        tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN",
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(list_corpus)

        # create sequence - h lista poy perieixe listes me lekseis (mia lista gia kathe perilhpsh) twra periexei listes me arithmoys.
        # kathe arithmos antistoixei se mia leksh me bash to leksiko poy dhmioyrghthke parapanw.
        sequence = tokenizer.texts_to_sequences(list_corpus)

        # padding sequence - kanoyme aytes tis listes me arithmoys na exoyn megethos 30 oles (ayto einai to megisto poy synantame).
        # stis perilhpseis opoy o arithmos tvn leksewn einai mikroteros sto telos ths listas mpainoyn mhdenika.
        X = kprocessing.sequence.pad_sequences(sequence, maxlen=30, padding="post", truncating="post")

        return X

    @staticmethod
    def preprocess_text(text, stopwords=None):
        # convert to lowercase and remove punctuations and characters
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

        # convert from string to list
        list_text = text.split()

        # remove Stopwords
        if stopwords is not None:
            list_text = [word for word in list_text if word not in stopwords]

        # back to string from list
        text = " ".join(list_text)

        return text
