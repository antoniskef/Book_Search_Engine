class Get:

    def __init__(self, es, search):
        self.es = es
        self.search = search
        self.res = "0"

    def get(self):

        query_body = {
                "query": {
                    "function_score": {
                        "query": {"query_string": {"query": "*{}*".format(self.search)}},
                        "functions": [
                            {
                                "field_value_factor": {
                                    "field": "rating",
                                    "factor": 10,
                                    "modifier": "sqrt",
                                    "missing": 1
                                }
                            },
                            {
                                "field_value_factor": {
                                    "field": "user_rating",
                                    "factor": 10,
                                    "modifier": "sqrt",
                                    "missing": 1
                                }
                            }
                        ],
                        # sum the score of each rating
                        "score_mode": "sum",
                        # sum above score with query score
                        "boost_mode": "sum"
                    }
                }
        }

        self.res = self.es.search(index="books", body=query_body, size=10000)

        print("Got %d Hits:" % self.res['hits']['total']['value'])

        for hit in self.res['hits']['hits']:
            print(hit["_source"])
