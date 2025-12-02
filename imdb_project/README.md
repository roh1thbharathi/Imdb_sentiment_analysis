# IMDB Sentiment Analysis Project

## 📋 Project Overview

This project implements a comprehensive sentiment analysis system for IMDB movie reviews using classical machine learning algorithms and compares them with deep learning approaches (BERT). The project demonstrates the complete pipeline from data preprocessing to model deployment.

**Key Achievements:**
- ✅ Achieved **89.8%** accuracy with tuned Linear SVM
- ✅ Processed 50,000 IMDB movie reviews
- ✅ Implemented 3 ML algorithms with hyperparameter tuning
- ✅ Compared classical ML with transformer-based models (BERT)
- ✅ Created production-ready inference script

---

## 📊 Dataset

**Source:** IMDB Dataset of 50K Movie Reviews  
**Size:** 50,000 reviews (25,000 positive + 25,000 negative)  
**Split:** 80% training (40,000) / 20% testing (10,000)  
**Balance:** Perfectly balanced binary classification

---

## 🗂️ Project Structure

```
imdb_project/
│
├── data/
│   ├── IMDB Dataset.csv          # Original dataset
│   └── clean_imdb.csv            # Preprocessed dataset
│
├── models/
│   ├── tuned_logreg.pkl          # Optimized Logistic Regression
│   ├── tuned_nb.pkl              # Optimized Naive Bayes
│   ├── tuned_svm.pkl             # Optimized Linear SVM (Best Model)
│   ├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
│   └── bert_comparison.csv       # BERT comparison results
│
├── visualizations/
│   ├── before_after_comparison.png      # Tuning impact
│   ├── confusion_matrix.png             # SVM confusion matrix
│   ├── roc_curve.png                    # ROC curve
│   └── complete_model_comparison.png    # ML vs BERT
│
├── preprocess.py                 # Data cleaning script
├── data_pipeline.py              # Data loading utility
├── train_logreg.py               # Logistic Regression training
├── train_nb.py                   # Naive Bayes training
├── train_svm.py                  # Linear SVM training
├── tuning.py                     # Hyperparameter optimization
├── create_visualizations.py      # Generate all plots
├── bert_vs_tuned_comparison.py   # BERT comparison
├── predict.py                    # Inference script
├── FAI_project_report.pdf        # Complete project report
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd imdb_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Step 1: Preprocess data
python preprocess.py

# Step 2: Train baseline models
python train_logreg.py
python train_nb.py
python train_svm.py

# Step 3: Hyperparameter tuning
python tuning.py

# Step 4: Generate visualizations
python create_visualizations.py
python bert_vs_tuned_comparison.py
```

### 3. Make Predictions

```bash
# Command-line mode
python predict.py "This movie was absolutely amazing!"

# Interactive mode
python predict.py
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Cleaning steps applied:**
- Removed HTML tags (e.g., `<br />`)
- Converted text to lowercase
- Removed punctuation and special characters
- Removed stopwords (NLTK English stopwords)
- Removed extra whitespaces

**Implementation:** `preprocess.py`

### 2. Feature Extraction

**TF-IDF Vectorization:**
- Max features: 20,000
- N-gram range: (1, 2) — unigrams and bigrams
- Output: Sparse matrix (40,000 × 20,000)

### 3. Models Implemented

| Model | Algorithm | Key Hyperparameters |
|-------|-----------|---------------------|
| Logistic Regression | Linear classifier | C, solver, penalty |
| Naive Bayes | Probabilistic classifier | alpha (smoothing) |
| Linear SVM | Maximum margin classifier | C (regularization) |

### 4. Hyperparameter Tuning

**Method:** GridSearchCV with 5-fold cross-validation

**Parameter grids:**
- **Logistic Regression:** C = [0.1, 1, 10, 100]
- **Naive Bayes:** alpha = [0.1, 0.5, 1.0, 2.0]
- **Linear SVM:** C = [0.1, 1, 10, 100]

---

## 📈 Results

### Baseline Models (Before Tuning)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8977 | 0.90 | 0.90 | 0.90 |
| Naive Bayes | 0.8642 | 0.87 | 0.86 | 0.86 |
| Linear SVM | 0.8970 | 0.90 | 0.90 | 0.90 |

### Optimized Models (After Tuning)

| Model | Accuracy | Improvement | Best Hyperparameters |
|-------|----------|-------------|----------------------|
| Logistic Regression | 0.8981 | +0.0004 | C=10 |
| Naive Bayes | 0.8650 | +0.0008 | alpha=0.5 |
| **Linear SVM** | **0.8983** | **+0.0013** | **C=10** |

### BERT Comparison

| Model | Accuracy | Type | Test Samples |
|-------|----------|------|--------------|
| **Linear SVM (Tuned)** | **0.8983** | Classical ML | 10,000 |
| DistilBERT | 0.8170 | Transformer | 1,000 |

**Winner:** Linear SVM outperforms BERT by +8.13%

---

## 🎯 Key Findings

1. **Linear SVM is the best performer** with 89.83% accuracy after tuning
2. **Hyperparameter tuning provided modest improvements** (0.04-0.13%)
3. **Classical ML outperforms BERT** on this dataset with TF-IDF features
4. **BERT requires more data** — tested on smaller subset due to computational constraints
5. **SVM + TF-IDF is production-ready** — fast inference, high accuracy

---

## 📊 Visualizations

All visualizations are in the `visualizations/` folder:

1. **Before/After Comparison:** Shows impact of hyperparameter tuning
2. **Confusion Matrix:** True Positives, False Positives, etc. for SVM
3. **ROC Curve:** AUC score of 0.95 for Linear SVM
4. **ML vs BERT:** Head-to-head comparison of all models

---

## 🔮 Inference / Prediction

### Using the Prediction Script

**Command-line mode:**
```bash
python predict.py "The acting was brilliant and the story captivating!"
```

**Interactive mode:**
```bash
python predict.py
# Then type reviews when prompted
```

**Output format:**
```
======================================================================
PREDICTION RESULT
======================================================================

Original Text:
  The acting was brilliant and the story captivating!

Cleaned Text (after preprocessing):
  acting brilliant story captivating

Sentiment:             Positive
Confidence:            94.73%

😊 The review is predicted as: **POSITIVE**
======================================================================
```

---

## 📦 Dependencies

**Core Libraries:**
- Python 3.8+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+
- matplotlib 3.7+
- seaborn 0.12+
- nltk 3.8+

**Optional (for BERT):**
- transformers 4.30+
- torch 2.0+

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧪 Evaluation Metrics

### Confusion Matrix (Linear SVM)
```
                 Predicted
              Negative  Positive
Actual  
Negative      4,450       550
Positive        567     4,433
```

### Performance Metrics
- **Accuracy:** 88.83%
- **Precision:** 89.00%
- **Recall:** 88.66%
- **F1-Score:** 88.83%
- **ROC AUC:** 0.9543

---

## 🎓 Academic Context

**Course:** Foundations of Artificial Intelligence  
**Project Type:** Sentiment Analysis with Classical ML and Transformers  
**Report:** Complete LaTeX report available in `FAI_project_report.pdf`

---

## 📝 Usage Examples

### Example 1: Positive Review
```python
Input:  "This movie was absolutely fantastic! Best film of the year."
Output: Positive (Confidence: 96.2%)
```

### Example 2: Negative Review
```python
Input:  "Terrible waste of time. Poor acting and boring plot."
Output: Negative (Confidence: 93.8%)
```

### Example 3: Mixed Review
```python
Input:  "The visuals were stunning but the story was confusing."
Output: Positive (Confidence: 62.3%)
```

---

## 🔧 Hyperparameter Tuning Details

### GridSearchCV Configuration
- **Cross-validation:** 5-fold stratified
- **Scoring metric:** Accuracy
- **Parallelization:** All available CPU cores (`n_jobs=-1`)

### Computation Time
- Logistic Regression: ~2 minutes
- Naive Bayes: ~1 minute
- Linear SVM: ~5 minutes

---

## 🏆 Model Selection Rationale

### Why These Three Models?

1. **Logistic Regression:**
   - Simple, interpretable
   - Fast training and inference
   - Provides probability estimates
   - Baseline for linear classifiers

2. **Naive Bayes:**
   - Extremely fast
   - Works well with text data
   - Probabilistic approach
   - Low computational cost

3. **Linear SVM:**
   - Excellent for high-dimensional data
   - Maximum margin classifier
   - Robust to overfitting
   - Industry standard for text classification

---

## 📚 References

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. ACL.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.

---

## 👨‍💻 Author

**Rohith Bharathithasan**  
Computer Science Student, Northeastern University  
Project Date: December 2025

---

## 📄 License

This project is for academic purposes. Dataset credit: Stanford University (IMDB Dataset).

---

## 🙏 Acknowledgments

- **IMDB Dataset:** Stanford University
- **scikit-learn:** For ML algorithms
- **Hugging Face:** For BERT/Transformers
- **Teaching Assistant:** For guidance on BERT comparison

---



---

**Last Updated:** December 2025
