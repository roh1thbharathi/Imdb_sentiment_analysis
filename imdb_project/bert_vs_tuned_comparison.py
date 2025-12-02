"""
BERT vs Tuned Models Comparison
Generates visualization comparing BERT with AFTER tuning results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

print("="*70)
print("BERT VS TUNED MODELS COMPARISON")
print("="*70)

# ============================================================================
# STEP 1: Load Test Data
# ============================================================================
print("\n[1/4] Loading test data...")
df = pd.read_csv('data/clean_imdb.csv')
X = df['clean_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Same split as training (must use same random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Test set: {len(X_test)} reviews")

# ============================================================================
# STEP 2: Load TF-IDF and Transform Test Data
# ============================================================================
print("\n[2/4] Loading TF-IDF vectorizer...")
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
X_test_tfidf = tfidf.transform(X_test)
print(f"   ✓ Test set vectorized: {X_test_tfidf.shape}")

# ============================================================================
# STEP 3: Load Tuned Models and Calculate Accuracies
# ============================================================================
print("\n[3/4] Loading tuned models and calculating accuracies...")

# Load models
with open('models/tuned_logreg.pkl', 'rb') as f:
    tuned_lr = pickle.load(f)
with open('models/tuned_nb.pkl', 'rb') as f:
    tuned_nb = pickle.load(f)
with open('models/tuned_svm.pkl', 'rb') as f:
    tuned_svm = pickle.load(f)

# Make predictions
y_pred_lr = tuned_lr.predict(X_test_tfidf)
y_pred_nb = tuned_nb.predict(X_test_tfidf)
y_pred_svm = tuned_svm.predict(X_test_tfidf)

# Calculate accuracies
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"   ✓ Tuned Logistic Regression: {acc_lr:.4f}")
print(f"   ✓ Tuned Naive Bayes: {acc_nb:.4f}")
print(f"   ✓ Tuned Linear SVM: {acc_svm:.4f}")

# ============================================================================
# STEP 4: Create Comparison Visualization with BERT
# ============================================================================
print("\n[4/4] Creating BERT vs Tuned Models comparison...")

# BERT result (from bert_comparison.csv)
bert_accuracy = 0.817  # DistilBERT on 1000 samples

# Prepare data
models = ['Logistic\nRegression\n(Tuned)', 'Naive Bayes\n(Tuned)', 
          'Linear SVM\n(Tuned)', 'DistilBERT\n(Transformer)']
accuracies = [acc_lr, acc_nb, acc_svm, bert_accuracy]
colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
model_types = ['Classical ML', 'Classical ML', 'Classical ML', 'Deep Learning']

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars
bars = ax.bar(models, accuracies, color=colors, alpha=0.85, 
              edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
           f'{acc:.4f}',
           ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add model type label at bottom of bar
    ax.text(bar.get_x() + bar.get_width()/2., 0.01,
           model_types[i],
           ha='center', va='bottom', fontsize=10, 
           style='italic', color='white', fontweight='bold')

# Add horizontal line for best classical model
best_classical = max(acc_lr, acc_nb, acc_svm)
ax.axhline(y=best_classical, color='green', linestyle='--', linewidth=2, 
          label=f'Best Classical ML: {best_classical:.4f}', alpha=0.7)

# Add horizontal line for BERT
ax.axhline(y=bert_accuracy, color='red', linestyle='--', linewidth=2, 
          label=f'BERT: {bert_accuracy:.4f}', alpha=0.7)

# Highlight winner
winner_idx = accuracies.index(max(accuracies))
bars[winner_idx].set_edgecolor('gold')
bars[winner_idx].set_linewidth(4)
ax.text(winner_idx, max(accuracies) + 0.02, '🏆 WINNER', 
       ha='center', fontsize=14, fontweight='bold', 
       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))

# Formatting
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison: Tuned Classical ML vs BERT\n' + 
            'IMDB Sentiment Analysis', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0.75, 0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)

# Add dataset info box
info_text = (
    "Dataset Information:\n"
    "• Classical ML: 10,000 test samples\n"
    "• BERT: 1,000 sample subset\n"
    "• TF-IDF features: 20,000 dims\n"
    "• After hyperparameter tuning"
)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
       fontsize=10, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add key findings
findings = (
    "Key Findings:\n"
    f"• Best Classical: Linear SVM ({acc_svm:.4f})\n"
    f"• BERT Accuracy: {bert_accuracy:.4f}\n"
    f"• Performance Gap: {abs(acc_svm - bert_accuracy):.4f}\n"
    "• Winner: Classical ML (Tuned SVM)"
)
ax.text(0.98, 0.98, findings, transform=ax.transAxes, 
       fontsize=10, verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()

# Save figure
output_path = 'visualizations/complete_model_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

# ============================================================================
# Print Summary
# ============================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\n{'Model':<30} {'Accuracy':<12} {'Type':<20}")
print("-" * 70)
print(f"{'Logistic Regression (Tuned)':<30} {acc_lr:.4f}       Classical ML")
print(f"{'Naive Bayes (Tuned)':<30} {acc_nb:.4f}       Classical ML")
print(f"{'Linear SVM (Tuned)':<30} {acc_svm:.4f}       Classical ML")
print(f"{'DistilBERT':<30} {bert_accuracy:.4f}       Deep Learning")
print("-" * 70)

winner = models[accuracies.index(max(accuracies))].replace('\n', ' ')
print(f"\n🏆 WINNER: {winner} with {max(accuracies):.4f} accuracy")

if max(accuracies) == acc_svm:
    print("\n✅ Classical ML (Tuned SVM) outperforms BERT!")
    print(f"   Performance advantage: +{acc_svm - bert_accuracy:.4f}")
else:
    print("\n✅ BERT outperforms Classical ML!")
    print(f"   Performance advantage: +{bert_accuracy - max(acc_lr, acc_nb, acc_svm):.4f}")

print("\n📊 Note: BERT tested on 1,000 samples, Classical ML on 10,000 samples")
print("="*70)
print("\n✓ Visualization updated with tuned model results!")
print("="*70)
