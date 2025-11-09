import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# 1. Load the raw CSV
csv_path = "data/IMDB Dataset.csv"
df = pd.read_csv(csv_path)

print("Original shape:", df.shape)

# One-time download for stopwords (safe to run multiple times)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercase
    - remove HTML tags
    - remove non-letter characters
    - remove extra spaces
    - remove common stopwords (optional but useful)
    """
    # lowercase
    text = text.lower()

    # remove html tags like <br />, <div>, etc
    text = re.sub(r"<.*?>", " ", text)

    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # split into words
    words = text.split()

    # remove stopwords (like 'the', 'is', 'and')
    words = [w for w in words if w not in stop_words]

    # join back into a single string
    return " ".join(words)

# 2. Apply cleaning to just a few rows to see the effect
print("\n=== Before cleaning (row 0) ===")
print(df["review"][0][:400])

cleaned_example = clean_text(df["review"][0])

print("\n=== After cleaning (row 0) ===")
print(cleaned_example[:400])

# 3. Show that we can create a new cleaned column
df["clean_review"] = df["review"].apply(clean_text)

print("\nNew columns:", df.columns.tolist())
print("Example clean_review (row 0):")
print(df["clean_review"][0][:400])

# 4. Convert sentiment text to numeric labels
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

print("\nLabel distribution (0 = negative, 1 = positive):")
print(df["label"].value_counts())

# 5. Save cleaned dataset
df.to_csv("data/clean_imdb.csv", index=False)
print("\nSaved cleaned dataset to data/clean_imdb.csv")
