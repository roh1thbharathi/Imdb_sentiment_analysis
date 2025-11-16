# train_svm.py
from data_pipeline import load_and_vectorize
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Get data + TF-IDF features
X_train_tfidf, X_test_tfidf, y_train, y_test, _ = load_and_vectorize()

print("Train shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)

# 2. Create and train Linear SVM model
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
print("Linear SVM training complete ✅")

# 3. Evaluate
svm_pred = svm_model.predict(X_test_tfidf)

svm_acc = accuracy_score(y_test, svm_pred)
print(f"\n[Linear SVM] Accuracy: {svm_acc:.4f}")
print("\nClassification report (Linear SVM):")
print(classification_report(y_test, svm_pred))
