# train_nb.py
from data_pipeline import load_and_vectorize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# 1. Get data + features (same pipeline)
X_train_tfidf, X_test_tfidf, y_train, y_test, _ = load_and_vectorize()

print("Train shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)

# 2. Create and train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
print("Naive Bayes training complete ✅")

# 3. Evaluate
nb_pred = nb_model.predict(X_test_tfidf)

nb_acc = accuracy_score(y_test, nb_pred)
print(f"\n[Naive Bayes] Accuracy: {nb_acc:.4f}")
print("\nClassification report (Naive Bayes):")
print(classification_report(y_test, nb_pred))
