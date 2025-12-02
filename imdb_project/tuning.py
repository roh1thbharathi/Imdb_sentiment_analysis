"""
Hyperparameter Tuning Script for IMDB Sentiment Analysis
Optimizes Logistic Regression, Naive Bayes, and Linear SVM
Compares before vs after tuning performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import time

print("="*70)
print("HYPERPARAMETER TUNING FOR IMDB SENTIMENT ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Load Preprocessed Data
# ============================================================================
print("\n[1/6] Loading preprocessed data...")
df = pd.read_csv('data/clean_imdb.csv')
print(f"   ✓ Loaded {len(df)} reviews")

# ============================================================================
# STEP 2: Train-Test Split (Same as original)
# ============================================================================
print("\n[2/6] Creating train-test split (80-20)...")
X = df['clean_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training set: {len(X_train)} reviews")
print(f"   ✓ Test set: {len(X_test)} reviews")

# ============================================================================
# STEP 3: TF-IDF Vectorization (Same as original)
# ============================================================================
print("\n[3/6] Applying TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"   ✓ Feature matrix shape: {X_train_tfidf.shape}")

# ============================================================================
# STEP 4: Store Original Results (from your previous runs)
# ============================================================================
print("\n[4/6] Recording baseline results...")
original_results = {
    'Logistic Regression': 0.8977,
    'Naive Bayes': 0.8642,
    'Linear SVM': 0.8970
}
print("   ✓ Baseline accuracies loaded")

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("   ✓ Created 'models/' directory")

# ============================================================================
# STEP 5: Hyperparameter Tuning
# ============================================================================
print("\n[5/6] Starting hyperparameter tuning...\n")

tuned_results = {}
best_models = {}

# ---------------------------------------------------------------------------
# 5.1: LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
print("─" * 70)
print("TUNING 1/3: LOGISTIC REGRESSION")
print("─" * 70)

print("Testing C values: [0.01, 0.1, 1, 10, 100]")
print("Running 5-fold cross-validation...")

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [1000]
}

start_time = time.time()
grid_lr = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_lr,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_lr.fit(X_train_tfidf, y_train)
elapsed_time = time.time() - start_time

# Get best model and evaluate
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_tfidf)
tuned_acc_lr = accuracy_score(y_test, y_pred_lr)

tuned_results['Logistic Regression'] = tuned_acc_lr
best_models['Logistic Regression'] = best_lr

print(f"\n✓ Tuning completed in {elapsed_time:.2f} seconds")
print(f"Best parameters: {grid_lr.best_params_}")
print(f"Best CV score: {grid_lr.best_score_:.4f}")
print(f"\nRESULTS:")
print(f"  Before tuning: {original_results['Logistic Regression']:.4f}")
print(f"  After tuning:  {tuned_acc_lr:.4f}")
print(f"  Improvement:   +{tuned_acc_lr - original_results['Logistic Regression']:.4f}")

# Save model
with open('models/tuned_logreg.pkl', 'wb') as f:
    pickle.dump(best_lr, f)
print("  Model saved: models/tuned_logreg.pkl")

# ---------------------------------------------------------------------------
# 5.2: NAIVE BAYES
# ---------------------------------------------------------------------------
print("\n" + "─" * 70)
print("TUNING 2/3: NAIVE BAYES")
print("─" * 70)

print("Testing alpha values: [0.1, 0.5, 1.0, 2.0, 5.0]")
print("Running 5-fold cross-validation...")

param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
}

start_time = time.time()
grid_nb = GridSearchCV(
    MultinomialNB(),
    param_grid_nb,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_nb.fit(X_train_tfidf, y_train)
elapsed_time = time.time() - start_time

# Get best model and evaluate
best_nb = grid_nb.best_estimator_
y_pred_nb = best_nb.predict(X_test_tfidf)
tuned_acc_nb = accuracy_score(y_test, y_pred_nb)

tuned_results['Naive Bayes'] = tuned_acc_nb
best_models['Naive Bayes'] = best_nb

print(f"\n✓ Tuning completed in {elapsed_time:.2f} seconds")
print(f"Best parameters: {grid_nb.best_params_}")
print(f"Best CV score: {grid_nb.best_score_:.4f}")
print(f"\nRESULTS:")
print(f"  Before tuning: {original_results['Naive Bayes']:.4f}")
print(f"  After tuning:  {tuned_acc_nb:.4f}")
print(f"  Improvement:   +{tuned_acc_nb - original_results['Naive Bayes']:.4f}")

# Save model
with open('models/tuned_nb.pkl', 'wb') as f:
    pickle.dump(best_nb, f)
print("  Model saved: models/tuned_nb.pkl")

# ---------------------------------------------------------------------------
# 5.3: LINEAR SVM
# ---------------------------------------------------------------------------
print("\n" + "─" * 70)
print("TUNING 3/3: LINEAR SVM")
print("─" * 70)

print("Testing C values: [0.1, 1, 10, 100]")
print("Running 5-fold cross-validation...")

param_grid_svm = {
    'C': [0.1, 1, 10, 100]
}

start_time = time.time()
grid_svm = GridSearchCV(
    LinearSVC(random_state=42, max_iter=2000),
    param_grid_svm,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_svm.fit(X_train_tfidf, y_train)
elapsed_time = time.time() - start_time

# Get best model and evaluate
best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test_tfidf)
tuned_acc_svm = accuracy_score(y_test, y_pred_svm)

tuned_results['Linear SVM'] = tuned_acc_svm
best_models['Linear SVM'] = best_svm

print(f"\n✓ Tuning completed in {elapsed_time:.2f} seconds")
print(f"Best parameters: {grid_svm.best_params_}")
print(f"Best CV score: {grid_svm.best_score_:.4f}")
print(f"\nRESULTS:")
print(f"  Before tuning: {original_results['Linear SVM']:.4f}")
print(f"  After tuning:  {tuned_acc_svm:.4f}")
print(f"  Improvement:   +{tuned_acc_svm - original_results['Linear SVM']:.4f}")

# Save model
with open('models/tuned_svm.pkl', 'wb') as f:
    pickle.dump(best_svm, f)
print("  Model saved: models/tuned_svm.pkl")

# ============================================================================
# STEP 6: Final Comparison Summary
# ============================================================================
print("\n" + "="*70)
print("FINAL COMPARISON: BEFORE vs AFTER TUNING")
print("="*70)

print(f"\n{'Model':<25} {'Before':<12} {'After':<12} {'Improvement':<15}")
print("-" * 70)

for model_name in original_results.keys():
    before = original_results[model_name]
    after = tuned_results[model_name]
    improvement = after - before
    improvement_pct = (improvement / before) * 100
    
    print(f"{model_name:<25} {before:.4f}       {after:.4f}       "
          f"+{improvement:.4f} ({improvement_pct:+.2f}%)")

print("-" * 70)

# Find best model
best_model_name = max(tuned_results, key=tuned_results.get)
best_accuracy = tuned_results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name} with {best_accuracy:.4f} accuracy")

# Save TF-IDF vectorizer for later use in prediction
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("\n✓ TF-IDF vectorizer saved: models/tfidf_vectorizer.pkl")

print("\n" + "="*70)
print("✓ HYPERPARAMETER TUNING COMPLETE!")
print("="*70)
print("\nAll optimized models saved in 'models/' directory:")
print("  - tuned_logreg.pkl")
print("  - tuned_nb.pkl")
print("  - tuned_svm.pkl")
print("  - tfidf_vectorizer.pkl")
