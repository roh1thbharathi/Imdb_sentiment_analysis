import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load cleaned dataset
df = pd.read_csv("data/clean_imdb.csv")

print("Full data shape:", df.shape)
print("Columns:", df.columns.tolist())


# 2. Define features (X) and labels (y)
X = df["clean_review"]   # model input: cleaned text
y = df["label"]          # model output: 0 = neg, 1 = pos

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 20% of data for testing
    random_state=42,     # fixed so results are reproducible
    stratify=y           # keeps 0/1 ratio similar in train and test
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# 4. Create TF-IDF vectorizer (turn text into numeric features)
vectorizer = TfidfVectorizer(
    max_features=20000  # limit vocabulary size so it doesn't explode
)


# Learn vocabulary + IDF from training data, then transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Use the same vocabulary to transform test data
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)
# Key ideas (super important):
# We call .fit_transform on X_train only:
# The model should not “peek” at test data while learning what words exist.
# We call .transform on X_test:
# Use the already learned vocabulary to encode test reviews.



# 5. Create Logistic Regression model
model = LogisticRegression(max_iter=1000)


# 6. Train (fit) the model on the training data
model.fit(X_train_tfidf, y_train)
print("Model training complete")


# 7. Predict on test data
y_pred = model.predict(X_test_tfidf)



# 8. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))