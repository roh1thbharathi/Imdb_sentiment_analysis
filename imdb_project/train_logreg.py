# train_logreg.py
from data_pipeline import load_and_vectorize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Get data + features
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_vectorize()

print("Train shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)

# 2. Create and train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Logistic Regression training complete ✅")

# 3. Evaluate
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print(f"\n[Logistic Regression] Accuracy: {acc:.4f}")
print("\nClassification report (Logistic Regression):")
print(classification_report(y_test, y_pred))


# 4. (Optional) Inspect important words
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

top_pos_idx = coef.argsort()[-20:]
top_neg_idx = coef.argsort()[:20]

print("\nTop 20 words for POSITIVE sentiment (LogReg):")
for i in reversed(top_pos_idx):
    print(f"{feature_names[i]:20s}  {coef[i]:.4f}")

print("\nTop 20 words for NEGATIVE sentiment (LogReg):")
for i in top_neg_idx:
    print(f"{feature_names[i]:20s}  {coef[i]:.4f}")
