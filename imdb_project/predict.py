"""
IMDB Sentiment Analysis - Prediction Script
Loads the tuned SVM model and makes real-time sentiment predictions
Usage: 
  python predict.py "Your review text here"
  python predict.py (for interactive mode)
"""

import pickle
import re
import sys
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================================================================
# Text Preprocessing (Same as training)
# ============================================================================

def preprocess_text(text):
    """
    Preprocesses input text the same way as training data:
    - Lowercase
    - Remove HTML tags
    - Remove punctuation and special characters
    - Remove stopwords
    - Remove extra spaces
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    # Join words back and remove extra spaces
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ============================================================================
# Load Models
# ============================================================================

def load_models():
    """
    Loads the tuned SVM model and TF-IDF vectorizer
    """
    try:
        print("Loading models...")
        
        # Load TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("  ✓ TF-IDF vectorizer loaded")
        
        # Load tuned SVM model (winner)
        with open('models/tuned_svm.pkl', 'rb') as f:
            model = pickle.load(f)
        print("  ✓ Tuned SVM model loaded")
        
        return vectorizer, model
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model files not found!")
        print(f"   Make sure you've run 'tuning.py' first to generate the models.")
        print(f"   Missing file: {e.filename}")
        sys.exit(1)

# ============================================================================
# Prediction Function
# ============================================================================

def predict_sentiment(text, vectorizer, model):
    """
    Predicts sentiment for a given text
    Returns: sentiment label and confidence score
    """
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    
    # Handle empty text after preprocessing
    if not cleaned_text:
        return "Unknown", 0.0, "Text too short or contains no meaningful words"
    
    # Transform using TF-IDF
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get confidence score (decision function gives distance from hyperplane)
    decision_score = model.decision_function(text_tfidf)[0]
    
    # Convert to probability-like confidence (using sigmoid-like transformation)
    confidence = 1 / (1 + np.exp(-abs(decision_score)))
    confidence = confidence * 100  # Convert to percentage
    
    # Map prediction to label
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence, cleaned_text

# ============================================================================
# Display Results
# ============================================================================

def display_result(original_text, sentiment, confidence, cleaned_text):
    """
    Displays prediction results in a nice format
    """
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    print(f"\nOriginal Text:")
    print(f"  {original_text}")
    print(f"\nCleaned Text (after preprocessing):")
    print(f"  {cleaned_text}")
    print(f"\n{'Sentiment:':<20} {sentiment}")
    print(f"{'Confidence:':<20} {confidence:.2f}%")
    
    # Add emoji for fun
    emoji = "😊" if sentiment == "Positive" else "😞"
    print(f"\n{emoji} The review is predicted as: **{sentiment.upper()}**")
    print("="*70)

# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function - handles command-line and interactive modes
    """
    print("="*70)
    print("IMDB SENTIMENT ANALYSIS - PREDICTION TOOL")
    print("="*70)
    
    # Load models
    vectorizer, model = load_models()
    print("\n✓ Models loaded successfully!\n")
    
    # Check if text provided as command-line argument
    if len(sys.argv) > 1:
        # Command-line mode
        review_text = ' '.join(sys.argv[1:])
        sentiment, confidence, cleaned = predict_sentiment(review_text, vectorizer, model)
        display_result(review_text, sentiment, confidence, cleaned)
    
    else:
        # Interactive mode
        print("="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("Type a movie review to analyze its sentiment.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            print("-"*70)
            review_text = input("\nEnter review: ").strip()
            
            # Check for exit commands
            if review_text.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the sentiment analyzer!")
                break
            
            # Check for empty input
            if not review_text:
                print("⚠️  Please enter some text.")
                continue
            
            # Make prediction
            sentiment, confidence, cleaned = predict_sentiment(review_text, vectorizer, model)
            display_result(review_text, sentiment, confidence, cleaned)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Usage examples:
    
    1. Command-line mode:
       python predict.py "This movie was absolutely amazing!"
       python predict.py "Worst film I've ever seen. Total waste of time."
    
    2. Interactive mode:
       python predict.py
       (then type reviews when prompted)
    """
    main()
