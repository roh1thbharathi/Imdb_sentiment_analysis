# data_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_vectorize(max_features: int = 20000):
    """
    Load cleaned IMDB data, split into train/test,
    and convert text to TF-IDF features.

    Returns:
        X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
    """
    # 1. Load cleaned CSV
    df = pd.read_csv("data/clean_imdb.csv")

    # 2. Define inputs and labels
    X = df["clean_review"]
    y = df["label"]

    # 3. Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit on train, transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
