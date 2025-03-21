# Sentiment Analysis: Classifying Customer Reviews Using Machine Learning

## Overview
This project develops a machine learning model to classify customer reviews as positive or negative. It demonstrates skills in text preprocessing, feature extraction, and model training, focusing on sentiment classification.

## Objective
- **Classify Reviews**: Predict sentiment (0 = negative, 1 = positive) from customer review text.

## Prerequisites
- **Python Version**: 3.7 or higher
- **Required Libraries**:
  - `tensorflow-datasets`
  - `nltk` (including `punkt`, `stopwords`)
  - `scikit-learn`
  - `pandas`
  - `numpy`
- **Installation Commands**:
  ```bash
  pip install tensorflow-datasets nltk scikit-learn pandas numpy
  python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
  ```

## File Structure
- **ML_Model.py**: Contains the full implementation of sentiment classification.

## Installation and Execution
### 1. Download the Code:
Save `ML_Model.py` to your local machine.

### 2. Install Dependencies:
Run the installation commands listed above.

### 3. Run the Script:
```bash
python ML_Model.py
```
The script loads the dataset, preprocesses text, extracts features, trains the model, and evaluates performance.

## Implementation Details
### Pipeline Steps
1. **Data Collection**: Loads IMDb Reviews dataset (25,000 labeled reviews).
2. **Preprocessing**:
   - Converts text to lowercase.
   - Removes digits and special characters.
   - Tokenizes using `nltk.word_tokenize`.
   - Removes stop words.
3. **Feature Extraction**:
   - Uses **TF-IDF Vectorization** (max 5000 features) for improved term importance analysis.
4. **Model Training**:
   - **Algorithm**: Logistic Regression.
   - **Split**: 80% train, 20% test.
   - **Tuning**: GridSearchCV optimizes model parameters.
5. **Evaluation**:
   - **Metrics**: Accuracy, Precision, Recall.
   - **Sample Predictions**: Displays actual vs predicted sentiment.

## Sample Output
```bash
Loading IMDb Reviews dataset...
Dataset size: 25000 reviews
Preprocessing text...
Extracting TF-IDF features...
Feature matrix shape: (20000, 5000)
Training Logistic Regression model...
Best Parameters: {'C': 1, 'solver': 'lbfgs'}
Evaluating model...
Accuracy: 87.5%
Precision: 88.0%
Recall: 87.0%

Sample Predictions:
Review: "Great film, loved every minute!"
True: 1 | Predicted: 1 (Positive)

Review: "Terrible, a waste of time."
True: 0 | Predicted: 0 (Negative)
```

## Validation
- **Test Split**: 20% test data (5,000 reviews)
- **Performance**:
  - Accuracy: ~87%
  - Balanced precision and recall, confirming reliable sentiment detection.

## Insights & Improvements
- **Why TF-IDF?**
  - Prioritizes meaningful terms, improving accuracy over Bag of Words.
- **Challenges**:
  - Context loss (e.g., negations like "not good" misclassified as positive).
- **Enhancements**:
  - Use **SVM** for improved classification.
  - Extend TF-IDF with bigram features (`ngram_range=(1,2)`).
  - Preserve negation context in preprocessing.

## Deliverables
- **Code**: `ML_Model.py`

## License
This project is developed for educational purposes and adheres to the assignment guidelines.
