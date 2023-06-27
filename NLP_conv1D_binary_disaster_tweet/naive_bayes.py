from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def naive_bayes():
    naive_model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB()),
         ])
    return naive_model