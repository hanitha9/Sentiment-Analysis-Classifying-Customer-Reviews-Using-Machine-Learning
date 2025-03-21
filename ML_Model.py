import re
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run once)
nltk.download('punkt')       # Base tokenizer
nltk.download('punkt_tab')   # Updated tokenizer data for newer NLTK versions
nltk.download('stopwords')

# Step 1: Data Collection
def load_data():
    """
    Load IMDb Reviews dataset from TensorFlow Datasets.
    
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns (0 = negative, 1 = positive).
    """
    # Load IMDb dataset (train split only for simplicity)
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True, split='train')
    # Convert to DataFrame
    data = [{'text': text.decode('utf-8'), 'label': label} for text, label in tfds.as_numpy(dataset)]
    return pd.DataFrame(data)

# Step 2: Data Preprocessing
def preprocess_text(text):
    """
    Clean and preprocess text for classification.
    
    Args:
        text (str): Raw review text.
    
    Returns:
        str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove digits and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back to text
    return ' '.join(tokens)

# Step 3: Feature Extraction
def extract_features(X_train, X_test):
    """
    Convert text to TF-IDF features.
    
    Args:
        X_train (list): Training text data.
        X_test (list): Testing text data.
    
    Returns:
        tuple: TF-IDF transformed training and testing data, and the vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Step 4: Model Selection and Training
def train_model(X_train_tfidf, y_train):
    """
    Train a Logistic Regression model with hyperparameter tuning.
    
    Args:
        X_train_tfidf (sparse matrix): TF-IDF features for training.
        y_train (array): Training labels.
    
    Returns:
        LogisticRegression: Trained model.
    """
    # Define model
    model = LogisticRegression(max_iter=1000)
    # Hyperparameter tuning with GridSearchCV
    param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_tfidf, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Step 5: Evaluation
def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the model using accuracy, precision, and recall.
    
    Args:
        model: Trained model.
        X_test_tfidf (sparse matrix): TF-IDF features for testing.
        y_test (array): Testing labels.
    """
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Main Execution
if __name__ == "__main__":
    # Load data
    print("Loading IMDb Reviews dataset...")
    df = load_data()
    print(f"Dataset size: {len(df)} reviews")

    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Split data (80/20)
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction
    print("Extracting TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    print(f"Feature matrix shape: {X_train_tfidf.shape}")

    # Train model
    print("Training Logistic Regression model...")
    model = train_model(X_train_tfidf, y_train)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test_tfidf, y_test)

    # Sample predictions with full preprocessed text
    print("\nSample Predictions (Full Preprocessed Text):")
    sample_reviews = X_test[:5].tolist()
    sample_labels = y_test[:5].tolist()
    sample_tfidf = X_test_tfidf[:5]
    predictions = model.predict(sample_tfidf)
    for review, true_label, pred_label in zip(sample_reviews, sample_labels, predictions):
        sentiment = "Positive" if pred_label == 1 else "Negative"
        print(f"Full Preprocessed Review: {review}")
        print(f"True: {true_label} | Predicted: {pred_label} ({sentiment})\n")
