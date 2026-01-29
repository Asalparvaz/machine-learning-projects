import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


TRANSLATOR = str.maketrans("", "", string.punctuation)

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.remove("not")

IGNORE_WORDS = {
    "film", "movie", "one", "even", "would", "time", "get", "story", "could",
    "plot", "make", "see", "also", "way", "little", "well", "people", "never",
    "know", "two", "another", "big", "made", "go", "back", "around", "going",
    "think", "still", "characters", "first", "character", "scene", "scenes",
    "films", "movies", "man", "new", "may", "take", "almost", "every", "things",
    "real", "comes", "come", "fact", "last", "point", "plays", "played", "role",
    "years", "john", "audience", "us",
}


def clean_text(text: str) -> str:

    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t.translate(TRANSLATOR) for t in tokens]
    tokens = [t for t in tokens if t]
    tokens = [
        t for t in tokens
        if t not in STOP_WORDS and t not in IGNORE_WORDS
    ]
    return " ".join(tokens)


def preprocess_documents(texts):
    return [clean_text(t) for t in texts]