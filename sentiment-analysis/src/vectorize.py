from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_tfidf_vectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )


def build_count_vectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    min_df=5,
):
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
