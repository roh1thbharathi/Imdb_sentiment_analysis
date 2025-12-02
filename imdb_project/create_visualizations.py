"""
Visualization Script for IMDB Sentiment Analysis
Generates 3 professional visualizations:
1. Before/After comparison bar chart
2. Confusion Matrix
3. ROC Curve

All outputs saved to visualizations/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import pickle
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*70)
print("CREATING VISUALIZATIONS FOR IMDB SENTIMENT ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Create Output Folder
# ============================================================================
print("\n[1/7] Setting up output directory...")
output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"   ✓ Created '{output_dir}/' folder")
else:
    print(f"   ✓ Using existing '{output_dir}/' folder")

# ============================================================================
# STEP 2: Load Data
# ============================================================================
print("\n[2/7] Loading preprocessed data...")
df = pd.read_csv('data/clean_imdb.csv')
print(f"   ✓ Loaded {len(df)} reviews")

# ============================================================================
# STEP 3: Recreate Train-Test Split (Same as tuning)
# ============================================================================
print("\n[3/7] Recreating train-test split...")
X = df['clean_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Test set: {len(X_test)} reviews")

# ============================================================================
# STEP 4: Load TF-IDF and Transform
# ============================================================================
print("\n[4/7] Loading TF-IDF vectorizer...")
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
X_test_tfidf = tfidf.transform(X_test)
print(f"   ✓ Test set vectorized: {X_test_tfidf.shape}")

# ============================================================================
# STEP 5: Load All Models and Get Predictions
# ============================================================================
print("\n[5/7] Loading tuned models and making predictions...")

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
from sklearn.metrics import accuracy_score
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"   ✓ Logistic Regression accuracy: {acc_lr:.4f}")
print(f"   ✓ Naive Bayes accuracy: {acc_nb:.4f}")
print(f"   ✓ Linear SVM accuracy: {acc_svm:.4f}")

# ============================================================================
# STEP 6: Create Visualizations
# ============================================================================
print("\n[6/7] Generating visualizations...\n")

# ---------------------------------------------------------------------------
# VISUALIZATION 1: Before/After Comparison Bar Chart
# ---------------------------------------------------------------------------
print("   Creating [1/3]: Before/After comparison...")

# Original results (from your previous runs)
original_results = {
    'Logistic\nRegression': 0.8977,
    'Naive\nBayes': 0.8642,
    'Linear\nSVM': 0.8970
}

# Tuned results
tuned_results = {
    'Logistic\nRegression': acc_lr,
    'Naive\nBayes': acc_nb,
    'Linear\nSVM': acc_svm
}

models = list(original_results.keys())
before_scores = [original_results[m] for m in models]
after_scores = [tuned_results[m] for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width/2, before_scores, width, label='Before Tuning', 
               color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, after_scores, width, label='After Tuning', 
               color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.2)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

add_labels(bars1)
add_labels(bars2)

# Add improvement arrows
for i, (before, after) in enumerate(zip(before_scores, after_scores)):
    improvement = after - before
    if improvement > 0:
        mid_x = i
        mid_y = (before + after) / 2
        ax.annotate('', xy=(mid_x + width/2, after - 0.001), 
                   xytext=(mid_x - width/2, before + 0.001),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(mid_x, mid_y, f'+{improvement:.4f}', 
               ha='center', va='center', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Formatting
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Model Performance: Before vs After Hyperparameter Tuning', 
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.set_ylim(0.85, 0.93)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/before_after_comparison.png', dpi=300, bbox_inches='tight')
print(f"      ✓ Saved: {output_dir}/before_after_comparison.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 2: Confusion Matrix (for best model - SVM)
# ---------------------------------------------------------------------------
print("   Creating [2/3]: Confusion matrix...")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Count'}, 
            square=True, linewidths=2, linecolor='black',
            annot_kws={'fontsize': 16, 'fontweight': 'bold'})

# Labels
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix - Tuned Linear SVM\n(Best Model)', 
            fontsize=15, fontweight='bold', pad=20)

# Set tick labels
ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=12, rotation=0)

# Add accuracy text
tn, fp, fn, tp = cm.ravel()
accuracy = (tn + tp) / (tn + fp + fn + tp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

stats_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, 
       fontsize=11, verticalalignment='center',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add explanation boxes
ax.text(0.25, -0.15, f'TN = {tn}\nCorrectly predicted\nNegative', 
       transform=ax.transAxes, ha='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(0.75, -0.15, f'FP = {fp}\nWrongly predicted\nPositive', 
       transform=ax.transAxes, ha='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.text(0.25, 1.08, f'FN = {fn}\nWrongly predicted\nNegative', 
       transform=ax.transAxes, ha='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.text(0.75, 1.08, f'TP = {tp}\nCorrectly predicted\nPositive', 
       transform=ax.transAxes, ha='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"      ✓ Saved: {output_dir}/confusion_matrix.png")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 3: ROC Curve (for best model - SVM)
# ---------------------------------------------------------------------------
print("   Creating [3/3]: ROC curve...")

# Get decision scores for ROC curve
if hasattr(tuned_svm, 'decision_function'):
    y_scores = tuned_svm.decision_function(X_test_tfidf)
else:
    y_scores = tuned_svm.predict_proba(X_test_tfidf)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve
ax.plot(fpr, tpr, color='#e74c3c', lw=3, 
       label=f'Linear SVM (AUC = {roc_auc:.4f})')

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
       label='Random Classifier (AUC = 0.5000)')

# Shade area under curve
ax.fill_between(fpr, tpr, alpha=0.2, color='#e74c3c')

# Formatting
ax.set_xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
ax.set_title('ROC Curve - Tuned Linear SVM\n(Receiver Operating Characteristic)', 
            fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

# Add interpretation text
interpretation = (
    f"AUC Interpretation:\n"
    f"• 0.90-1.00 = Excellent\n"
    f"• 0.80-0.90 = Good\n"
    f"• 0.70-0.80 = Fair\n"
    f"• 0.60-0.70 = Poor\n"
    f"• 0.50-0.60 = Fail"
)
ax.text(0.98, 0.02, interpretation, transform=ax.transAxes, 
       fontsize=9, verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Mark optimal point (closest to top-left corner)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
       label=f'Optimal Point (threshold={optimal_threshold:.3f})')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
print(f"      ✓ Saved: {output_dir}/roc_curve.png")
plt.close()

# ============================================================================
# STEP 7: Generate Summary Report
# ============================================================================
print("\n[7/7] Generating summary statistics...\n")

print("="*70)
print("VISUALIZATION SUMMARY")
print("="*70)

print("\n📊 BEFORE vs AFTER TUNING:")
print("-"*70)
for i, model in enumerate(['Logistic Regression', 'Naive Bayes', 'Linear SVM']):
    before = list(original_results.values())[i]
    after = list(tuned_results.values())[i]
    improvement = after - before
    print(f"{model:20} | Before: {before:.4f} | After: {after:.4f} | +{improvement:.4f}")

print("\n🏆 BEST MODEL: Linear SVM")
print("-"*70)
print(f"Accuracy:  {acc_svm:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

print("\n📈 CONFUSION MATRIX BREAKDOWN:")
print("-"*70)
print(f"True Negatives (TN):  {tn:5d}  ✓ Correctly predicted negative")
print(f"False Positives (FP): {fp:5d}  ✗ Wrongly predicted positive")
print(f"False Negatives (FN): {fn:5d}  ✗ Wrongly predicted negative")
print(f"True Positives (TP):  {tp:5d}  ✓ Correctly predicted positive")
print(f"\nTotal Correct:   {tn + tp:5d} / {len(y_test)}")
print(f"Total Incorrect: {fp + fn:5d} / {len(y_test)}")

print("\n📁 OUTPUT FILES:")
print("-"*70)
print(f"✓ {output_dir}/before_after_comparison.png")
print(f"✓ {output_dir}/confusion_matrix.png")
print(f"✓ {output_dir}/roc_curve.png")

print("\n" + "="*70)
print("✓ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*70)
print(f"\nAll plots saved in '{output_dir}/' folder")
print("Ready to include in your report! 📄")
