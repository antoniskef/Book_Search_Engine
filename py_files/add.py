import csv
from elasticsearch import helpers
import pandas as pd


class Add:
    def __init__(self, es, userid):
        self.es = es
        self.userid = userid
        self.df_with_av = pd.DataFrame
        self.df_final = pd.DataFrame
        self.valid = 0
        self.with_review = 0

    def add_elastic(self):
        self.es.indices.delete(index="books", ignore=[400, 404])

        self.es.indices.create(index="books", ignore=[400, 404])

        self.es.indices.put_mapping(
            index="books",
            body={
                "properties": {
                    "isbn": {"type": "text"},
                    "book_title": {"type": "text"},
                    "book_author": {"type": "text"},
                    " year_of_publication": {"type": "text"},
                    "publisher": {"type": "text"},
                    "summary": {"type": "text"},
                    "category": {"type": "text"},
                    "rating": {"type": "integer"},
                    "user_rating": {"type": "integer"}
                }
            }
        )

        with open('data/Final_BX-Books.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(self.es, reader, index="books")

    def add_average_ratings(self):
        df_books = pd.read_csv('data/BX-Books.csv')

        df_ratings = pd.read_csv('data/BX-Book-Ratings.csv')
        df_ratings = df_ratings.drop("uid", axis=1)

        df_ratings = df_ratings.groupby(['isbn'], as_index=False).mean()

        self.df_with_av = pd.merge(df_books, df_ratings, how='inner', on=['isbn'])

    def add_user_ratings(self):
        df_ratings = pd.read_csv('data/BX-Book-Ratings.csv')
        df_ratings = df_ratings[df_ratings['uid'].astype(str) == self.userid]

        df_ratings = df_ratings.drop("uid", axis=1)

        df_ratings = df_ratings.rename(columns={'rating': 'user_rating'})

        self.df_final = pd.merge(self.df_with_av, df_ratings, how='left', on=['isbn'])

        self.df_final.to_csv('data/Final_BX-Books.csv', index=False)

    def valid_user(self):
        df_users = pd.read_csv('data/BX-Users.csv')

        df_users = df_users[df_users['uid'].astype(str) == self.userid]

        if len(df_users) == 0:
            self.valid = 0
        else:
            self.valid = 1

    def user_with_review(self):
        df_users = pd.read_csv('data/BX-Book-Ratings.csv')
        df_users = df_users[df_users['uid'].astype(str) == self.userid]

        if len(df_users) == 0:
            self.with_review = 0
        else:
            self.with_review = 1




